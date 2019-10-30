// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Linear layer.

#include "linearlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


LinearLayer::LinearLayer(int in_features, int out_features, bool bias)
: Layer() {
	layer_type_id_ = 2;
	in_features_ = in_features;
	out_features_ = out_features;
	init_range_ = sqrt(1.0 / in_features_);
	w_ = Rand(out_features_, in_features_, 1, 1, init_range_, -init_range_);
	dw_ = Zeros(w_);
	has_weight_ = true;
	has_bias_ = bias;
	if (has_bias_) {
		b_ = Rand(out_features_, 1, 1, 1, init_range_, -init_range_);
		db_ = Zeros(b_);
	}
}


LinearLayer::~LinearLayer() {
}


Tensor LinearLayer::Forward(const Tensor& input_image) {
	if (has_bias_) {
		int batch_size = input_image.cols();
		return MM(w_, input_image) + Repmat(b_, 1, batch_size, 1, 1);
	}
	else {
		return MM(w_, input_image);
	}
}


Tensor LinearLayer::Backward(const Tensor& gradient) {
	if (is_first_layer_) {
		dw_ += MM(gradient, Transpose(input_));
	}
	else {
		dw_ += MM(gradient, Transpose(previous_layer_->GetOutput()));
	}
	if (has_bias_) {
		db_ += Sum(gradient, 1);
	}
	return MM(Transpose(w_), gradient);
}


void LinearLayer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	int32_t param_i[3];
	float param_f[1];
	param_i[0] = (int32_t)in_features_;
	param_i[1] = (int32_t)out_features_;
	param_i[2] = (int32_t)has_bias_;
	param_f[0] = init_range_;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)param_i, sizeof(int32_t) * 3);
	output_file.write((char*)param_f, sizeof(float) * 1);
	WriteTensor(output_file, &w_);
	WriteTensor(output_file, &b_);
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void LinearLayer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	int32_t param_i[3];
	float param_f[1];
	input_file.read((char*)param_i, sizeof(int32_t) * 3);
	input_file.read((char*)param_f, sizeof(float) * 1);
	in_features_ = (int)param_i[0];
	out_features_ = (int)param_i[1];
	has_bias_ = param_i[2]? 1: 0;
	init_range_ = param_f[0];
	ReadTensor(input_file, &w_);
	ReadTensor(input_file, &b_);
	input_file.read((char *)&end_of_layer, sizeof(int32_t));
	if (END_OF_LAYER != end_of_layer) {
		cerr << "Error: Invalid model format." << endl;
		exit(1);
	}
	dw_ = Zeros(w_);
	if (has_bias_) {
		db_ = Zeros(b_);
	}
}
