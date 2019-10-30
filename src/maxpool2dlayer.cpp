// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Max pooling 2D layer.

#include "maxpool2dlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


MaxPool2dLayer::MaxPool2dLayer(int kernel_size)
: Layer() {
	layer_type_id_ = 11;
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


void MaxPool2dLayer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	int32_t param_i[1];
	param_i[0] = (int32_t)kernel_size_;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)param_i, sizeof(int32_t) * 1);
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void MaxPool2dLayer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	int32_t param_i[1];
	input_file.read((char*)param_i, sizeof(int32_t) * 1);
	kernel_size_ = (int)param_i[0];
	input_file.read((char *)&end_of_layer, sizeof(int32_t));
	if (END_OF_LAYER != end_of_layer) {
		cerr << "Error: Invalid model format." << endl;
		exit(1);
	}
}
