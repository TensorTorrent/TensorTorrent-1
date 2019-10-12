// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Adam optimizer.

#include "adam.h"


using namespace ftensor;


Adam::Adam(Sequential& model, float lr, float beta1, float beta2, float eps)
: Optimizer(model) {
	lr_ = lr;
	beta1_ = beta1;
	beta2_ = beta2;
	eps_ = eps;
}


Adam::~Adam() {
}


void Adam::Step() {
}
