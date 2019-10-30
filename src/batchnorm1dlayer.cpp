// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Batch normalization layer for 1-D cases.

#include "batchnorm1dlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


BatchNorm1dLayer::BatchNorm1dLayer(int num_features, float eps, float momentum)
: Layer() {
	layer_type_id_ = 7;
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


BatchNorm1dLayer::~BatchNorm1dLayer() {
}


Tensor BatchNorm1dLayer::Forward(const Tensor& input) {
	xi_ = input;
	batch_size_ = input.cols();
	if (training_mode_) {
		e_input_ = Mean(input, 1);
		var_input_ = Var(input, 1, "0");
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
	return Mul(Div(input - r_e_input_, Sqrt(r_var_input_ + eps_)), r_w_) + r_b_;
}


Tensor BatchNorm1dLayer::Backward(const Tensor& gradient) {
	Tensor dldxh = Mul(gradient, r_w_);
	Tensor dldob2 = Sum(Mul(Mul(-0.5 * dldxh, xi_ - r_e_input_), Pow(r_var_input_ + eps_, -1.5)), 1);
	Tensor dldub = Sum(Mul(dldxh, Div(-1.0, Sqrt(r_var_input_ + eps_))), 1) + Mul(dldob2, Sum(-2.0 * (xi_ - r_e_input_), 1));
	Tensor dldxi = Mul(dldxh, Div(1.0, Sqrt(r_var_input_ + eps_))) + Mul(Repmat(dldob2, 1, batch_size_, 1, 1), Mul(xi_ - r_e_input_, 2.0 / batch_size_)) + Mul(Repmat(dldub, 1, batch_size_, 1, 1), 1.0 / batch_size_);
	dw_ = Sum(Mul(gradient, output_), 1) / (1.0 * batch_size_);
	db_ = Sum(gradient, 1) / (1.0 * batch_size_);
	return dldxi;
}


void BatchNorm1dLayer::ExportTo(std::ofstream& output_file) {
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


void BatchNorm1dLayer::ImportFrom(std::ifstream& input_file) {
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
