// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer.

#include "conv2dlayer.h"


using namespace ftensor;
using std::endl;
using std::cerr;


Conv2dLayer::Conv2dLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
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
		b_ = Rand(1, 1, 1, 1, init_range_, -init_range_);
	}
}


Conv2dLayer::~Conv2dLayer() {
}


Tensor Conv2dLayer::Forward(const Tensor& input) {
	return Conv2d(input, w_, stride_, padding_);
}


Tensor Conv2dLayer::Backward(const Tensor& gradient) {
	auto grad = gradient;
	Tensor in;
	for (int i_kernel = 0; i_kernel < out_channels_; ++i_kernel) {
		for (int i_channel = 0; i_channel < in_channels_; ++i_channel) {
			if (is_first_layer_) {
				in = input_;
			}
			else {
				in = previous_layer_->GetOutput();
			}
			Tensor dw_x = dw_.S(-1, -1, -1, -1, i_channel, i_channel + 1, i_kernel, i_kernel + 1) + Conv2d(Permute(in.S(-1, -1, -1, -1, i_channel, i_channel + 1), 0, 1, 3, 2), Permute(grad.S(-1, -1, -1, -1, i_kernel, i_kernel + 1), 0, 1, 3, 2), stride_, padding_);
			dw_.S(dw_x, -1, -1, -1, -1, i_channel, i_channel + 1, i_kernel, i_kernel + 1);
		}
	}
	auto back_kernel = Permute(w_, 0, 1, 3, 2);
	return ConvTranspose2d(grad, back_kernel, stride_, padding_);
}
