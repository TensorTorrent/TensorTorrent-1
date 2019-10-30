// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: Leaky ReLU layer.

#include "leakyrelulayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


LeakyReluLayer::LeakyReluLayer(float negative_slope)
: Layer() {
	layer_type_id_ = 12;
	negative_slope_ = negative_slope;
}


LeakyReluLayer::~LeakyReluLayer() {
}


Tensor LeakyReluLayer::Forward(const Tensor& input) {
	output_ = Where(input >= 0, input, negative_slope_ * input);
	return output_;
}


Tensor LeakyReluLayer::Backward(const Tensor& gradient) {
	grad_output_ = Where(output_ >= 0, gradient, negative_slope_ * gradient);
	return grad_output_;
}


void LeakyReluLayer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	float param_f[1];
	param_f[0] = negative_slope_;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)param_f, sizeof(float) * 1);
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void LeakyReluLayer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	float param_f[1];
	input_file.read((char*)param_f, sizeof(float) * 1);
	negative_slope_ = param_f[0];
	input_file.read((char *)&end_of_layer, sizeof(int32_t));
	if (END_OF_LAYER != end_of_layer) {
		cerr << "Error: Invalid model format." << endl;
		exit(1);
	}
}
