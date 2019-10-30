// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer.

#include "conv2dlayer.h"


using namespace ftensor;
using std::endl;
using std::cerr;


Conv2dLayer::Conv2dLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
: Layer() {
	layer_type_id_ = 5;
	in_channels_ = in_channels;
	out_channels_ = out_channels;
	kernel_size_ = kernel_size;
	stride_ = stride;
	padding_ = padding;
	init_range_ = sqrt(1.0 / in_channels_ / kernel_size_ / kernel_size_);
	has_weight_ = true;
	w_ = Rand(kernel_size_, kernel_size_, in_channels_, out_channels_, init_range_, -init_range_);
	dw_ = Zeros(w_);
	has_bias_ = bias;
	if (has_bias_) {
		b_ = Rand(1, 1, out_channels_, 1, init_range_, -init_range_);
		db_ = Zeros(b_);
	}
}


Conv2dLayer::~Conv2dLayer() {
}


Tensor Conv2dLayer::Forward(const Tensor& input) {
	batch_size_ = input.gros();
	if (has_bias_) {
		Tensor temp = Conv2d(input, w_, stride_, padding_);
		return temp + Repmat(b_, temp.rows(), temp.cols(), 1, temp.gros());
	}
	else {
		return Conv2d(input, w_, stride_, padding_);
	}
}


Tensor Conv2dLayer::Backward(const Tensor& gradient) {
	Tensor in;
	auto grad = gradient;
	if (has_bias_) {
		for (int i_kernel = 0; i_kernel < out_channels_; ++i_kernel) {
			db_(0, 0, i_kernel, 0) += Sum(grad.S(-1, -1, -1, -1, i_kernel, i_kernel + 1, -1, -1)).Item() / (1.0 * batch_size_);
		}
	}
	for (int i_kernel = 0; i_kernel < out_channels_; ++i_kernel) {
		for (int i_channel = 0; i_channel < in_channels_; ++i_channel) {
			if (is_first_layer_) {
				in = input_;
			}
			else {
				in = previous_layer_->GetOutput();
			}
			Tensor dw_x = dw_.S(-1, -1, -1, -1, i_channel, i_channel + 1, i_kernel, i_kernel + 1) + Conv2d(Permute(in.S(-1, -1, -1, -1, i_channel, i_channel + 1), 0, 1, 3, 2), Permute(grad.S(-1, -1, -1, -1, i_kernel, i_kernel + 1), 0, 1, 3, 2), stride_, padding_) / (1.0 * batch_size_);
			dw_.S(dw_x, -1, -1, -1, -1, i_channel, i_channel + 1, i_kernel, i_kernel + 1);
		}
	}
	auto back_kernel = Permute(w_, 0, 1, 3, 2);
	auto output_temp = ConvTranspose2d(grad, back_kernel, stride_, padding_);
	if (is_first_layer_) {
		if (Match(output_temp, input_)) {
			return output_temp;
		}
		else {
			auto output_temp2 = Zeros(input_);
			int rows_o = output_temp.rows();
			int cols_o = output_temp.cols();
			int slis_o = output_temp.slis();
			int gros_o = output_temp.gros();
			output_temp2.S(output_temp, 0, rows_o, 0, cols_o, 0, slis_o, 0, gros_o);
			return output_temp2;
		}
	}
	else {
		if (Match(output_temp, previous_layer_->GetOutput())) {
			return output_temp;
		}
		else {
			auto output_temp2 = Zeros(previous_layer_->GetOutput());
			int rows_o = output_temp.rows();
			int cols_o = output_temp.cols();
			int slis_o = output_temp.slis();
			int gros_o = output_temp.gros();
			output_temp2.S(output_temp, 0, rows_o, 0, cols_o, 0, slis_o, 0, gros_o);
			return output_temp2;
		}
	}
}


void Conv2dLayer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	int32_t param_i[6];
	float param_f[1];
	param_i[0] = (int32_t)in_channels_;
	param_i[1] = (int32_t)out_channels_;
	param_i[2] = (int32_t)kernel_size_;
	param_i[3] = (int32_t)stride_;
	param_i[4] = (int32_t)padding_;
	param_i[5] = (int32_t)has_bias_;
	param_f[0] = init_range_;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)param_i, sizeof(int32_t) * 6);
	output_file.write((char*)param_f, sizeof(float) * 1);
	WriteTensor(output_file, &w_);
	WriteTensor(output_file, &b_);
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void Conv2dLayer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	int32_t param_i[6];
	float param_f[1];
	input_file.read((char*)param_i, sizeof(int32_t) * 6);
	input_file.read((char*)param_f, sizeof(float) * 1);
	in_channels_ = (int)param_i[0];
	out_channels_ = (int)param_i[1];
	kernel_size_ = (int)param_i[2];
	stride_ = (int)param_i[3];
	padding_ = (int)param_i[4];
	has_bias_ = param_i[5]? 1: 0;
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
