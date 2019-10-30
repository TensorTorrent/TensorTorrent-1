// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Flatten layer.

#include "flattenlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


FlattenLayer::FlattenLayer(int start_dim, int end_dim)
: Layer() {
	layer_type_id_ = 13;
	start_dim_ = start_dim;
	end_dim_ = end_dim;
	shape_[0] = 0;
	shape_[1] = 0;
	shape_[2] = 0;
	shape_[3] = 0;
}


FlattenLayer::~FlattenLayer() {
}


Tensor FlattenLayer::Forward(const Tensor& input) {
	shape_[0] = input.rows();
	shape_[1] = input.cols();
	shape_[2] = input.slis();
	shape_[3] = input.gros();
	output_ = Transpose(Reshape(input, shape_[3], shape_[0] * shape_[1] * shape_[2], 1, 1));
	return output_;
}


Tensor FlattenLayer::Backward(const Tensor& gradient) {
	grad_output_ = Reshape(Transpose(gradient), shape_[0], shape_[1], shape_[2], shape_[3]);
	return grad_output_;
}


void FlattenLayer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	int32_t param_i[2];
	param_i[0] = (int32_t)start_dim_;
	param_i[1] = (int32_t)end_dim_;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)param_i, sizeof(int32_t) * 2);
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void FlattenLayer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	int32_t param_i[2];
	input_file.read((char*)param_i, sizeof(int32_t) * 2);
	start_dim_ = (int)param_i[0];
	end_dim_ = (int)param_i[1];
	input_file.read((char *)&end_of_layer, sizeof(int32_t));
	if (END_OF_LAYER != end_of_layer) {
		cerr << "Error: Invalid model format." << endl;
		exit(1);
	}
}
