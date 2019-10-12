#include <iostream>
#include <ctime>
#include "tensortorrent.h"


using namespace ftensor;
using std::string;
using std::vector;
using std::cout;
using std::endl;


const int kBatchSize = 50;
const int kNumEpochs = 1;


int main() {
	// Load the MNIST database
	ftensor::Tensor training_images, training_labels, testing_images, testing_labels;
	LoadDatabase("/opt/Datasets/MNIST", training_images, training_labels, testing_images, testing_labels);

	// Define layers
	FlattenLayer f1;
	LinearLayer l2(784, 400, false);
	ReluLayer r3;
	LinearLayer l4(400, 200, false);
	ReluLayer r5;
	LinearLayer l6(200, 100, false);
	ReluLayer r7;
	LinearLayer l8(100, 10, false);
	SoftmaxLayer s9;

	// Construct the model
	Sequential model ({&f1, &l2, &r3, &l4, &r5, &l6, &r7, &l8, &s9});

	auto c = model(testing_images);
	auto d = c * 0 + 1;
	auto e = model.Backward(d);
	c.Info();
	e.Info();
	
	return 0;
}
