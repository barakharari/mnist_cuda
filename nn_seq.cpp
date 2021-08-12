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

void
swap(int &i) {
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
		float upstream_gradient(int i, int batch_i);
		void back_prop(int batch_i);
	public:
		OutputLayer *next_layer;
		InputLayer *prev_layer;
		float nodes[FCSIZE];
		float bias[FCSIZE];
		float bias_derivs[FCSIZE][BSIZE];
		float weights[FCSIZE][ISIZE];
		float weight_derivs[FCSIZE][ISIZE][BSIZE];
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
		float nodes[OSIZE];
		float bias[OSIZE];
		float bias_derivs[FCSIZE][BSIZE];
		float weights[OSIZE][FCSIZE];
		float weight_derivs[OSIZE][FCSIZE][BSIZE];
};

InputLayer::InputLayer(){}

FullyConnectedLayer::FullyConnectedLayer(){
	for (int i = 0; i < FCSIZE; i++){
		for (int j = 0; j < ISIZE; j++){
			this->weights[i][j] = distribution(generator);
		}
		this->bias[i] = 0;
	}
}

OutputLayer::OutputLayer(){
	for (int i = 0; i < OSIZE; i++){
		for (int j = 0; j < FCSIZE; j++){
			this->weights[i][j] = 0; 
		}
		this->bias[i] = 0;
	}
}

int
InputLayer::predict(float* inp){
	this->nodes = inp;
	return this->next_layer->predict();
}

void fp_fc_seq(float nodes[FCSIZE], float inps[ISIZE], float weights[FCSIZE][ISIZE], float bias[FCSIZE]){
	for (int i = 0; i < FCSIZE; i++){
		nodes[i] = 0;
	//sum weights*inp
		for (int j = 0; j < ISIZE; j++){
			nodes[i] += weights[i][j]*inps[j];	
		} 
	//bias
		nodes[i] += bias[i];
	//relu
		nodes[i] = std::max((float)0, nodes[i]);
	}
}

int
FullyConnectedLayer::predict(){
	fp_fc_seq(this->nodes, this->prev_layer->nodes, this->weights, this->bias);
	return this->next_layer->predict();
}

int
OutputLayer::predict(){

	float* inps = this->prev_layer->nodes;

	//classification
	int prediction = 0;
	for (int i = 0; i < OSIZE; i++){
		this->nodes[i] = 0;
		//sum weights*inp
		for (int j = 0; j < FCSIZE; j++){
			this->nodes[i] += this->weights[i][j]*inps[j];	
		} 
		//bias
		this->nodes[i] += this->bias[i];
		if (this->nodes[i] > this->nodes[prediction]){
			prediction = i;
		}
	}	
	return prediction;
}

void
InputLayer::train(float* inps, int label, int batch_i){
	this->nodes = inps;
	this->next_layer->train(label, batch_i);
}

void
FullyConnectedLayer::train(int label, int batch_i){
	fp_fc_seq(this->nodes, this->prev_layer->nodes, this->weights, this->bias);
	this->next_layer->train(label, batch_i);
}

void
OutputLayer::train(int label, int batch_i){

	float* inps = this->prev_layer->nodes;

	float denom = 0; //denom for softmax

	//update node values
	for (int i = 0; i < OSIZE; i++){

		this->nodes[i] = 0;
		//sum weights*inp
		for (int j = 0; j < FCSIZE; j++) this->nodes[i] += this->weights[i][j]*inps[j];	

//		std::cout << this->nodes[i] << std::endl;
 		
		//bias
		this->nodes[i] += this->bias[i];

		//summing up exp for softmax denom
		denom += exp(this->nodes[i]);
	}
//	std::cout << std::endl;

	
	//softmax
	for (int i = 0; i < OSIZE; i++){
		this->nodes[i] = (exp(this->nodes[i]) / denom);
		if (isnan(this->nodes[i])) this->nodes[i] = 100;
		if (this->nodes[i] != this->nodes[i]){
			std::cout << exp(this->nodes[i]) << " / " << denom << std::endl;
		}
	}
	this->back_prop(label, batch_i);
}

void
OutputLayer::back_prop(int label, int batch_i){

	for (int i = 0; i < OSIZE; i++){
		float l_wtr_oi = i == label ? this->nodes[i] - 1 : this->nodes[i];
		for (int j = 0; j < FCSIZE; j++){
			//deriv of Input to output layer with respect to weight of current edge
			// times deriv of Softmax output with respect to Input to softmax
			// time deriv of cost func with respect to softmax output
			
			float oi_r_w = this->prev_layer->nodes[j];
			this->weight_derivs[i][j][batch_i] = l_wtr_oi * oi_r_w; 
		}
		this->bias_derivs[i][batch_i] = l_wtr_oi;
	}	
	this->prev_layer->back_prop(batch_i);
}

float
FullyConnectedLayer::upstream_gradient(int i, int batch_i){
	float o_deriv = 0;
	for (int j = 0; j < OSIZE; j++){
		o_deriv += this->next_layer->weight_derivs[j][i][batch_i];
		std::cout << o_deriv << " ";
	}
	std::cout << std::endl;
	return o_deriv;
}

void 
FullyConnectedLayer::back_prop(int batch_i){
	for (int i = 0; i < FCSIZE; i++){
		float o_deriv = 0;
		for (int j = 0; j < OSIZE; j++){
			o_deriv += this->next_layer->weight_derivs[j][i][batch_i];
		}
		if (this->nodes[i] == 0){
			for (int j = 0; j < ISIZE; j++)	this->weight_derivs[i][j][batch_i] = 0;
			this->bias_derivs[i][batch_i] = 0;
		} else{
			for (int j = 0; j < ISIZE; j++) this->weight_derivs[i][j][batch_i] = this->prev_layer->nodes[j] * o_deriv;
			this->bias_derivs[i][batch_i] = o_deriv;	
		}
	}
}

void
InputLayer::update_weights(float learning_rate){
	this->next_layer->update_weights(learning_rate);
}

void
FullyConnectedLayer::update_weights(float learning_rate){

	for (int i = 0; i < FCSIZE; i++){

		//update bias
		float avg_b = this->bias_derivs[i][0];
		for (int j = 1; j < BSIZE; j++)	avg_b += this->bias_derivs[i][j];
	
		avg_b /= BSIZE;
		this->bias[i] -= learning_rate * avg_b;

		for (int j = 0; j < ISIZE; j++){
			float avg_g = this->weight_derivs[i][j][0];
			for (int k = 1; k < BSIZE; k++) avg_g += this->weight_derivs[i][j][k];
			if (avg_g) avg_g /= BSIZE;
			this->weights[i][j] -= learning_rate * avg_g; 
		}
	}
	this->next_layer->update_weights(learning_rate);	
}

void
OutputLayer::update_weights(float learning_rate){

	for (int i = 0; i < OSIZE; i++){

		//update bias
		float avg_b = this->bias_derivs[i][0];
		for (int j = 1; j < BSIZE; j++){
			avg_b += this->bias_derivs[i][j];
		}

		avg_b /= BSIZE;
		this->bias[i] -= learning_rate * avg_b;

		//update weights associated with node i
		for (int j = 0; j < FCSIZE; j++){
			float avg_g = this->weight_derivs[i][j][0];
			for (int k = 1; k < BSIZE; k++) avg_g += this->weight_derivs[i][j][k]; 
			avg_g /= BSIZE;
			this->weights[i][j] -= learning_rate * avg_g; 
		}
	}

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
