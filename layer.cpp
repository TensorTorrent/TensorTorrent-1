// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Layers.

#include "layer.h"


Layer::Layer() {
	is_first_layer_ = true;
	previous_layer_ = nullptr;
}


Layer::~Layer() {
}


void Layer::Forward() {
	if (is_first_layer_) {
		output_ = Forward(input_);
	}
	else {
		output_ = Forward(previous_layer_->GetOutput());
	}
}


void Layer::Backward() {
	grad_output_ = Backward(grad_input_);
	if (!is_first_layer_) {
		previous_layer_->SetGradInput(grad_output_);
	}
}


void Layer::ZeroGrad() {
}
