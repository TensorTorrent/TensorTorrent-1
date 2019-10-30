// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Batch normalization layer for 2-D cases.

#include "batchnorm2dlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


BatchNorm2dLayer::BatchNorm2dLayer(int num_features, float eps, float momentum)
: Layer() {
	layer_type_id_ = 8;
	num_features_ = num_features;
	eps_ = eps;
	momentum_ = momentum;
	batch_size_ = 1;
	has_weight_ = true;
	has_bias_ = true;
	w_ = Ones(num_features_, 1);
	dw_ = Zeros(num_features_, 1);
	b_ = Zeros(num_features_, 1);
	db_ = Zeros(num_features_, 1);
	e_input_ = Zeros(num_features_, 1);
	var_input_ = Zeros(num_features_, 1);
	h_e_input_ = Zeros(num_features_, 1);
	h_var_input_ = Ones(num_features_, 1);
}


BatchNorm2dLayer::~BatchNorm2dLayer() {
}


Tensor BatchNorm2dLayer::Forward(const Tensor& input) {
	shape_[0] = input.rows();
	shape_[1] = input.cols();
	shape_[2] = input.slis();
	shape_[3] = input.gros();
	xi_ = Transpose(Reshape(Permute(input, 0, 1, 3, 2), shape_[0] * shape_[1] * shape_[3], shape_[2], 1, 1));
	batch_size_ = xi_.cols();
	if (training_mode_) {
		e_input_ = Mean(xi_, 1);
		var_input_ = Var(xi_, 1, "0");
		h_e_input_ = (1.0 - momentum_) * h_e_input_ + momentum_ * e_input_;
		h_var_input_ = (1.0 - momentum_) * h_var_input_ + momentum_ * var_input_;
	}
	else {
		e_input_ = h_e_input_;
		var_input_ = h_var_input_;
	}
	r_e_input_ = Repmat(e_input_, 1, batch_size_, 1, 1);
	r_var_input_ = Repmat(var_input_, 1, batch_size_, 1, 1);
	r_w_ = Repmat(w_, 1, batch_size_, 1, 1);
	r_b_ = Repmat(b_, 1, batch_size_, 1, 1);
	return Permute(Reshape(Transpose(Mul(Div(xi_ - r_e_input_, Sqrt(r_var_input_ + eps_)), r_w_) + r_b_), shape_[0], shape_[1], shape_[3], shape_[2]), 0, 1, 3, 2);
}


Tensor BatchNorm2dLayer::Backward(const Tensor& gradient) {
	Tensor grad = Transpose(Reshape(Permute(gradient, 0, 1, 3, 2), shape_[0] * shape_[1] * shape_[3], shape_[2], 1, 1));
	Tensor output_temp = Transpose(Reshape(Permute(output_, 0, 1, 3, 2), shape_[0] * shape_[1] * shape_[3], shape_[2], 1, 1));
	Tensor dldxh = Mul(grad, r_w_);
	Tensor dldob2 = Sum(Mul(Mul(-0.5 * dldxh, xi_ - r_e_input_), Pow(r_var_input_ + eps_, -1.5)), 1);
	Tensor dldub = Sum(Mul(dldxh, Div(-1.0, Sqrt(r_var_input_ + eps_))), 1) + Mul(dldob2, Sum(-2.0 * (xi_ - r_e_input_), 1));
	Tensor dldxi = Mul(dldxh, Div(1.0, Sqrt(r_var_input_ + eps_))) + Mul(Repmat(dldob2, 1, batch_size_, 1, 1), Mul(xi_ - r_e_input_, 2.0 / batch_size_)) + Mul(Repmat(dldub, 1, batch_size_, 1, 1), 1.0 / batch_size_);
	dw_ = Sum(Mul(grad, output_temp), 1) / (1.0 * batch_size_);
	db_ = Sum(grad, 1) / (1.0 * batch_size_);
	return Permute(Reshape(Transpose(dldxi), shape_[0], shape_[1], shape_[3], shape_[2]), 0, 1, 3, 2);
}


void BatchNorm2dLayer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	int32_t param_i[1];
	float param_f[2];
	param_i[0] = (int32_t)num_features_;
	param_f[0] = eps_;
	param_f[1] = momentum_;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)param_i, sizeof(int32_t) * 1);
	output_file.write((char*)param_f, sizeof(float) * 2);
	WriteTensor(output_file, &w_);
	WriteTensor(output_file, &b_);
	WriteTensor(output_file, &h_e_input_);
	WriteTensor(output_file, &h_var_input_);
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void BatchNorm2dLayer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	int32_t param_i[1];
	float param_f[2];
	input_file.read((char*)param_i, sizeof(int32_t) * 1);
	input_file.read((char*)param_f, sizeof(float) * 2);
	num_features_ = (int)param_i[0];
	eps_ = param_f[0];
	momentum_ = param_f[1];
	ReadTensor(input_file, &w_);
	ReadTensor(input_file, &b_);
	ReadTensor(input_file, &h_e_input_);
	ReadTensor(input_file, &h_var_input_);
	input_file.read((char *)&end_of_layer, sizeof(int32_t));
	if (END_OF_LAYER != end_of_layer) {
		cerr << "Error: Invalid model format." << endl;
		exit(1);
	}
	dw_ = Zeros(num_features_, 1);
	db_ = Zeros(num_features_, 1);
}
