// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Softmax layer.

#include "softmaxlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;

const float EXP_LIMIT = 85.0;


SoftmaxLayer::SoftmaxLayer(int dim)
: Layer() {
	layer_type_id_ = 4;
	if (dim >= 0 && dim <=3){
		dim_ = dim;
	}
	else {
		cerr << "Error: Dimension out of range." << endl;
		exit(1);
	}
}


SoftmaxLayer::~SoftmaxLayer() {
}


Tensor SoftmaxLayer::Forward(const Tensor& input) {
	Tensor temp = Exp(Where(input > EXP_LIMIT, input, input));
	Tensor sum = Sum(temp, dim_);
	switch (dim_) {
		case 0:
		output_ = Div(temp, Repmat(sum, input.rows(), 1, 1, 1));
		break;
		case 1:
		output_ = Div(temp, Repmat(sum, 1, input.cols(), 1, 1));
		break;
		case 2:
		output_ = Div(temp, Repmat(sum, 1, 1, input.slis(), 1));
		break;
		case 3:
		output_ = Div(temp, Repmat(sum, 1, 1, 1, input.gros()));
		break;
		default:
		;
	}
	return output_;
}


Tensor SoftmaxLayer::Backward(const Tensor& gradient) {
	grad_output_ = gradient;
	return grad_output_;
}


void SoftmaxLayer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	int32_t param_i[1];
	param_i[0] = (int32_t)dim_;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)param_i, sizeof(int32_t) * 1);
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void SoftmaxLayer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	int32_t param_i[1];
	input_file.read((char*)param_i, sizeof(int32_t) * 1);
	dim_ = (int)param_i[0];
	input_file.read((char *)&end_of_layer, sizeof(int32_t));
	if (END_OF_LAYER != end_of_layer) {
		cerr << "Error: Invalid model format." << endl;
		exit(1);
	}
}
