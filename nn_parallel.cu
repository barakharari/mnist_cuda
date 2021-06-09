#include <chrono>
#include <cmath>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

#define ISIZE 28*28
#define FCSIZE 1024 
#define OSIZE 10
#define BSIZE 1 

class InputLayer;
class FullyConnectedLayer;
class OutputLayer;

std::random_device generator;
std::uniform_real_distribution<float> distribution(-1,1);

#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
void
gpu_assert_h(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void
swap (int &i) {
    // Some of the & are superfluous.
    i =
     (0xff&(i >> 24)) |
     (0xff00&(i >> 8)) |
     (0xff0000&(i << 8)) |
     (0xff000000&(i << 24));
}

int
read_int(int fd) {
    int rv;
    int i;
    rv = read(fd, &i, 4); assert(rv == 4);
    swap(i);
    return i;
}

template <int N>
void
read_mnist_images(const std::string &fn, float (&imgs)[N][28*28]) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n_images = read_int(fd);
    assert(n_images == N);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    for (int i = 0; i < N; i++) {
        unsigned char tmp[ISIZE];
        rv = read(fd, tmp, ISIZE); assert(rv == ISIZE);
        for (int r = 0; r < ISIZE; r++) {
	    imgs[i][r] = float(tmp[r])/127.5 - 1; 
        }
    }

    rv = close(fd); assert(rv == 0);
}

template <int N>
void
read_mnist_labels(const std::string &fn, unsigned char (&labels)[N]) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n_labels = read_int(fd);
    assert(n_labels == N);

    rv = read(fd, labels, N); assert(rv == N);
    #if 0
    for (int i = 0; i < N; i++) {
        assert(labels[i] >= 0 && labels[i] <= 9);
    }
    #endif

    rv = close(fd); assert(rv == 0);
}


class InputLayer{
	public:
		int predict(float* inps);
		void update_weights(float learning_rate);
		void train(float *inp, int label, int batch_i);
		InputLayer();
	public:
		float *nodes;
		FullyConnectedLayer *next_layer;
};

class FullyConnectedLayer{
	public:
		FullyConnectedLayer();
		int predict();
		void train(int label, int batch_i);
		void update_weights(float learning_rate);
		void back_prop(int batch_i);
	public:
		OutputLayer *next_layer;
		InputLayer *prev_layer;
		float *nodes;
		float *bias;
		float *weights;
		float *bias_derivs;
		float *weight_derivs;
};

class OutputLayer{
	public:
		OutputLayer();
		int predict();
		void train(int label, int batch_i);
		void update_weights(float learning_rate);
		void back_prop(int label, int batch_i);
	public:
		FullyConnectedLayer *prev_layer;
		float *nodes;
		float *bias;
		float *weights;
		float *bias_derivs;
		float *weight_derivs;
};

InputLayer::InputLayer(){
	cudaError_t rv_ce;
	rv_ce = cudaMalloc(&this->nodes, ISIZE*sizeof(float));
	gpu_assert(rv_ce);
}

FullyConnectedLayer::FullyConnectedLayer(){
	cudaError_t rv_ce;
	rv_ce = cudaMalloc(&this->nodes, FCSIZE*sizeof(float));
	gpu_assert(rv_ce);

	float weights[FCSIZE * ISIZE];
	float bias[FCSIZE];

	for (int i = 0; i < FCSIZE * ISIZE; i++){
		weights[i] = distribution(generator);
	}
	for (int i = 0; i < FCSIZE; i++){
		bias[i] = 0;
	}	

	rv_ce = cudaMalloc(&this->bias_derivs, FCSIZE*BSIZE*sizeof(float));	
	gpu_assert(rv_ce);

	rv_ce = cudaMalloc(&this->weight_derivs, FCSIZE*ISIZE*BSIZE*sizeof(float));
	gpu_assert(rv_ce);

	rv_ce = cudaMalloc(&this->weights, FCSIZE*ISIZE*sizeof(float));
	gpu_assert(rv_ce);	

	rv_ce = cudaMalloc(&this->bias, FCSIZE*sizeof(float));	
	gpu_assert(rv_ce);

	{
		rv_ce = cudaMemcpy(this->bias, bias, FCSIZE*sizeof(float), cudaMemcpyHostToDevice);
		gpu_assert(rv_ce);
		rv_ce = cudaMemcpy(this->weights, weights, FCSIZE*ISIZE*sizeof(float), cudaMemcpyHostToDevice);
		gpu_assert(rv_ce);
	}
	std::cout << "Initialized FC" << std::endl;
}

OutputLayer::OutputLayer(){
	cudaError_t rv_ce;
	rv_ce = cudaMalloc(&this->nodes, OSIZE*sizeof(float));
	gpu_assert(rv_ce);

	float weights[OSIZE * FCSIZE];
	float bias[OSIZE];

	for (int i = 0; i < OSIZE * FCSIZE; i++){
		weights[i] = 0;
	}
	
	for (int i = 0; i < OSIZE; i++){
		bias[i] = 0;
	}

	rv_ce = cudaMalloc(&this->bias_derivs, OSIZE*BSIZE*sizeof(float));	
	gpu_assert(rv_ce);

	rv_ce = cudaMalloc(&this->weight_derivs, OSIZE*FCSIZE*BSIZE*sizeof(float));
	gpu_assert(rv_ce);

	rv_ce = cudaMalloc(&this->weights, OSIZE*FCSIZE*sizeof(float));
	gpu_assert(rv_ce);	

	rv_ce = cudaMalloc(&this->bias, OSIZE*sizeof(float));
	gpu_assert(rv_ce);

	{
		rv_ce = cudaMemcpy(this->bias, bias, OSIZE*sizeof(float), cudaMemcpyHostToDevice);
		gpu_assert(rv_ce);
		rv_ce = cudaMemcpy(this->weights, weights, OSIZE*FCSIZE*sizeof(float), cudaMemcpyHostToDevice);
		gpu_assert(rv_ce);
	}
	std::cout << "Initialized OL" << std::endl;
}

//MARK:: forward_pass FC
__global__ void
fp_cuda(float *nodes, float *inps, float *weights, float *bias, int relu, int prev_layer_size){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	nodes[id] = 0;
	//sum weights*inp
	float* w_p = weights + (id * prev_layer_size);
	for (int j = 0; j < prev_layer_size; j++){
		nodes[id] += w_p[j]*inps[j];	
	} 
	//bias
	nodes[id] += bias[id];

	//relu
	if (relu) nodes[id] = max((float)0, nodes[id]);
}

int
InputLayer::predict(float* inps){

	//Copy input to device
	cudaError_t rv_ce;
	rv_ce = cudaMemcpy(this->nodes, inps, ISIZE*sizeof(float), cudaMemcpyHostToDevice); 
	gpu_assert(rv_ce);

	return this->next_layer->predict();
}

int
FullyConnectedLayer::predict(){
	fp_cuda<<<1, FCSIZE>>>(this->nodes, this->prev_layer->nodes, this->weights, this->bias, 1, ISIZE);
	return this->next_layer->predict();
}

int
OutputLayer::predict(){

	fp_cuda<<<1, OSIZE>>>(this->nodes, this->prev_layer->nodes, this->weights, this->bias, 0, FCSIZE);

	//Copy output to host 
	float nodes[OSIZE];

	cudaError_t rv_ce;
	rv_ce = cudaMemcpy(nodes, this->nodes, OSIZE*sizeof(float), cudaMemcpyDeviceToHost); 
	gpu_assert(rv_ce);

	int prediction = 0;
	for (int i = 0; i < OSIZE; i++){
		if (nodes[i] > nodes[prediction]){
			prediction = i;
		}
	}
	return prediction;
}

void
InputLayer::train(float* inps, int label, int batch_i){

	//Copy input to device
	cudaError_t rv_ce;
	rv_ce = cudaMemcpy(this->nodes, inps, ISIZE*sizeof(float), cudaMemcpyHostToDevice); 
	gpu_assert(rv_ce);

	this->next_layer->train(label, batch_i);
}

void
FullyConnectedLayer::train(int label, int batch_i){
	fp_cuda<<<1, FCSIZE>>>(this->nodes, this->prev_layer->nodes, this->weights, this->bias, 1, ISIZE);
	this->next_layer->train(label, batch_i);
}

void
OutputLayer::train(int label, int batch_i){

	fp_cuda<<<1, OSIZE>>>(this->nodes, this->prev_layer->nodes, this->weights, this->bias, 0, FCSIZE);		

	float nodes[OSIZE];

	cudaError_t rv_ce;
	rv_ce = cudaMemcpy(nodes, this->nodes, OSIZE*sizeof(float), cudaMemcpyDeviceToHost); 
	gpu_assert(rv_ce);

	float denom = 0; //denom for softmax
	for (int i = 0; i < OSIZE; i++)	denom += exp(nodes[i]);
	
	//softmax
	for (int i = 0; i < OSIZE; i++){
		nodes[i] = exp(nodes[i]) / denom;
		if (isnan(nodes[i])) nodes[i] = 100;
		if (nodes[i] != nodes[i]) std::cout << exp(nodes[i]) << " / " << denom << std::endl;
	}

	rv_ce = cudaMemcpy(this->nodes, nodes, OSIZE*sizeof(float), cudaMemcpyHostToDevice); 
	gpu_assert(rv_ce);

	this->back_prop(label, batch_i);
}

__global__
void ol_bp_cuda(float *nodes, float* prev_nodes, int out_node_i, int batch_i, float l_wtr_oi, float* weight_derivs, float *bias_derivs){

	int id = blockIdx.x*blockDim.x + threadIdx.x;

	float oi_r_w = prev_nodes[id];
	float *wd = weight_derivs + (FCSIZE * BSIZE * out_node_i) + (BSIZE * id) + (batch_i);
	float *bd = bias_derivs + (BSIZE * out_node_i) + batch_i; 
	*wd = l_wtr_oi * oi_r_w; 
	*bd = l_wtr_oi;

}

void
OutputLayer::back_prop(int label, int batch_i){

	float nodes[OSIZE];

	cudaError_t rv_ce;
	rv_ce = cudaMemcpy(nodes, this->nodes, OSIZE*sizeof(float), cudaMemcpyDeviceToHost); 
	gpu_assert(rv_ce);

	for (int i = 0; i < OSIZE; i++){

		float l_wtr_oi = i == label ? nodes[i] - 0.999 : nodes[i] + 0.001;
		ol_bp_cuda<<<1, FCSIZE>>>(this->nodes, this->prev_layer->nodes, i, batch_i, l_wtr_oi, this->weight_derivs, this->bias_derivs);	
		
		#if 0
		for (int j = 0; j < FCSIZE; j++){
			//deriv of Input to output layer with respect to weight of current edge
			// times deriv of Softmax output with respect to Input to softmax
			// time deriv of cost func with respect to softmax output
			
			float oi_r_w = this->prev_layer->nodes[j];
			this->weight_derivs[i][j][batch_i] = l_wtr_oi * oi_r_w; 

		}
		this->bias_derivs[i][batch_i] = l_wtr_oi;

		#endif
	}	
	this->prev_layer->back_prop(batch_i);

}

__global__
void fc_bp_cuda(float *nodes, float *prev_nodes, float *b_derivs, float *next_w_derivs, float *w_derivs, int batch_i){

	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	float o_deriv = 0;
	for (int j = 0; j < OSIZE; j++){
		o_deriv += *(next_w_derivs + (FCSIZE * BSIZE * j) + (BSIZE * id) + batch_i);
	}
	if (nodes[id] == 0){
		for (int j = 0; j < ISIZE; j++){
			float *w = w_derivs + (FCSIZE * BSIZE * id) + (BSIZE * j) + batch_i; 
			*w = 0;
		}
		float* b = b_derivs + (BSIZE * id) + batch_i;
		*b = 0;
	} else{
		for (int j = 0; j < ISIZE; j++){
			float *w = w_derivs + (FCSIZE * BSIZE * id) + (BSIZE * j) + batch_i; 
			*w = prev_nodes[j] * o_deriv;
		}
		float* b = b_derivs + (BSIZE * id) + batch_i;
		*b = o_deriv;
	}
}

void 
FullyConnectedLayer::back_prop(int batch_i){
	fc_bp_cuda<<<1, FCSIZE>>>(this->nodes, this->prev_layer->nodes, this->bias_derivs, this->next_layer->weight_derivs, this->weight_derivs, batch_i);
}

void
InputLayer::update_weights(float learning_rate){
	this->next_layer->update_weights(learning_rate);
}

__global__
void fc_update_weights(float learning_rate, float *bias_derivs, float *bias, float *weight_derivs, float *weights, int prev_layer_size){
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	//update bias
	float avg_b = *(bias_derivs + (BSIZE * id));
	for (int j = 1; j < BSIZE; j++)	avg_b += *(bias_derivs + (BSIZE*id) + j);
	
	avg_b /= BSIZE;
	bias[id] -= learning_rate * avg_b;

	for (int j = 0; j < prev_layer_size; j++){
		float avg_g = *(weight_derivs + (prev_layer_size * BSIZE * id) + (BSIZE * j));
		for (int k = 1; k < BSIZE; k++) avg_g += *(weight_derivs + (prev_layer_size * BSIZE * id) + (BSIZE * j) + k);
		avg_g /= BSIZE;
		float* w = weights + (prev_layer_size * id) + j;
		*w -= learning_rate * avg_g; 
	}
}

void
FullyConnectedLayer::update_weights(float learning_rate){
	fc_update_weights<<<1, FCSIZE>>>(learning_rate, this->bias_derivs, this->bias, this->weight_derivs, this->weights, ISIZE);
	this->next_layer->update_weights(learning_rate);	
}

void
OutputLayer::update_weights(float learning_rate){
	fc_update_weights<<<1, FCSIZE>>>(learning_rate, this->bias_derivs, this->bias, this->weight_derivs, this->weights, FCSIZE);
}

int main(){

    static float training_images[60000][ISIZE];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);

    static unsigned char training_labels[60000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59999] == 8);

    static float test_images[10000][ISIZE];
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    static unsigned char test_labels[10000];
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    InputLayer il = InputLayer();
    FullyConnectedLayer fl = FullyConnectedLayer();
    OutputLayer ol = OutputLayer();

    il.next_layer = &fl;
    fl.next_layer = &ol;
    fl.prev_layer = &il;
    ol.prev_layer = &fl;

    std::default_random_engine eng;
    std::uniform_int_distribution<size_t> pick_test(0, 9999);

    auto start = std::chrono::high_resolution_clock::now();

//20 epochs or runs through data set
    for (int e = 0; e < 20; e++) {

        // Create shuffled sequence of training images.
        std::vector<int> training(60000);
        std::iota(training.begin(), training.end(), 0);
        assert(*--training.end() == 59999);
        std::shuffle(training.begin(), training.end(), eng);

//Batches of 600
        for (int r = 0; r < 60000/BSIZE; r++) {

//Get progress at round
            if (r%1000 == 0) {

                // fprintf(stderr, "Begin predict...."); fflush(stderr);
                int correct = 0;
                for (size_t i = 0; i < 100; i++) {
                    // fprintf(stderr, "Predict: %d for %lu\n", il.predict(training_images[i]), i);
                    size_t ind = pick_test(eng);
                    if (il.predict(test_images[ind]) == test_labels[ind]) {
                        correct++;
                    }
                }
                fprintf(stderr, "Epoch %d: Round %d: accuracy=%f\n", e, r, correct/100.0);
            }

//Train with 100 more images
            for (size_t i = 0; i < BSIZE; i++) {
                il.train(training_images[training.at(BSIZE*r + i)], training_labels[training.at(BSIZE*r + i)], i);
            }
//Learning rate .002
            il.update_weights(0.0002);
        }

    std::chrono::duration<double, std::milli> dt = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Duration: " << dt.count() << std::endl;    

    }
}
