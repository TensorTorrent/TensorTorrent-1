// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Softmax layer.

#include "softmaxlayer.h"


using namespace ftensor;
using std::cerr;
using std::endl;

const float SOFTMAX_LIMIT = 85.0;


SoftmaxLayer::SoftmaxLayer(int dim)
: Layer() {
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
	Tensor temp = Exp(Where(input > SOFTMAX_LIMIT, input, input));
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
