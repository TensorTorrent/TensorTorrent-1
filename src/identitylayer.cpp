// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Identity layer.

#include "identitylayer.h"


using namespace ftensor;


IdentityLayer::IdentityLayer()
: Layer() {
}


IdentityLayer::~IdentityLayer() {
}


Tensor IdentityLayer::Forward(const Tensor& input) {
	return input;
}


Tensor IdentityLayer::Backward(const Tensor& gradient) {
	return gradient;
}
