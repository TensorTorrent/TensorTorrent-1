// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Transposed convolution layer.

#include "convtranspose2dlayer.h"


using namespace ftensor;
using std::endl;
using std::cerr;


ConvTranspose2dLayer::ConvTranspose2dLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
: Layer() {
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


ConvTranspose2dLayer::~ConvTranspose2dLayer() {
}


Tensor ConvTranspose2dLayer::Forward(const Tensor& input) {
	if (has_bias_) {
		Tensor temp = ConvTranspose2d(input, w_, stride_, padding_);
		return temp + Repmat(b_, temp.rows(), temp.cols(), 1, temp.gros());
	}
	else {
		return ConvTranspose2d(input, w_, stride_, padding_);
	}
}


Tensor ConvTranspose2dLayer::Backward(const Tensor& gradient) {
	Tensor in;
	auto grad = gradient;
	if (has_bias_) {
		for (int i_kernel = 0; i_kernel < out_channels_; ++i_kernel) {
			db_(0, 0, i_kernel, 0) += Sum(grad.S(-1, -1, -1, -1, i_kernel, i_kernel + 1, -1, -1)).Item();
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
			Tensor kron_temp = Zeros(stride_, stride_, 1, 1);
			kron_temp(0, 0, 0, 0) = 1;
			auto temp = PaddingAsym(Kron(Permute(in.S(-1, -1, -1, -1, i_channel, i_channel + 1), 0, 1, 3, 2), kron_temp), kernel_size_ - 1, kernel_size_ - stride_, kernel_size_ - 1, kernel_size_ - stride_);
			Tensor dw_x = dw_.S(-1, -1, -1, -1, i_channel, i_channel + 1, i_kernel, i_kernel + 1) + Rot90(Conv2d(temp, Padding(Permute(grad.S(-1, -1, -1, -1, i_kernel, i_kernel + 1), 0, 1, 3, 2), padding_, padding_, 0, 0), 1, 0), 2);
			dw_.S(dw_x, -1, -1, -1, -1, i_channel, i_channel + 1, i_kernel, i_kernel + 1);
		}
	}
	auto back_kernel = Permute(w_, 0, 1, 3, 2);
	auto output_temp = Conv2d(grad, Rot90(back_kernel, 2), stride_, padding_);
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
