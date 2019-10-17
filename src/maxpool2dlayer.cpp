// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Max pooling 2D layer.

#include "maxpool2dlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


MaxPool2dLayer::MaxPool2dLayer(int kernel_size)
: Layer() {
	if (kernel_size > 0) {
		kernel_size_ = kernel_size;
	}
	else {
		cerr << "Error: Invalid kernel size." << endl;
	}
}


MaxPool2dLayer::~MaxPool2dLayer() {
}


Tensor MaxPool2dLayer::Forward(const Tensor& input) {
	return MaxPool2d(input, kernel_size_, &mask_);
}


Tensor MaxPool2dLayer::Backward(const Tensor& gradient) {
	Tensor temp = Kron(gradient, Ones(kernel_size_, kernel_size_, 1, 1));
	if (is_first_layer_) {
		if (Match(temp, input_)) {
			return Mul(mask_, temp);
		}
		else {
			int rows_o = temp.rows();
			int cols_o = temp.cols();
			int slis_o = temp.slis();
			int gros_o = temp.gros();
			Tensor output_temp = Zeros(input_);
			output_temp.S(temp, 0, rows_o, 0, cols_o, 0, slis_o, 0, gros_o);
			return Mul(mask_, output_temp);
		}
	}
	else {
		if (Match(temp, previous_layer_->GetOutput())) {
			return Mul(mask_, temp);
		}
		else {
			int rows_o = temp.rows();
			int cols_o = temp.cols();
			int slis_o = temp.slis();
			int gros_o = temp.gros();
			Tensor output_temp = Zeros(previous_layer_->GetOutput());
			output_temp.S(temp, 0, rows_o, 0, cols_o, 0, slis_o, 0, gros_o);
			return Mul(mask_, output_temp);
		}
	}
}
