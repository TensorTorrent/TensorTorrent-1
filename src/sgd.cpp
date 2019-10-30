// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: SGD optimizer.

#include "sgd.h"


using namespace ftensor;


SGD::SGD(Model& model, float lr, float momentum)
: Optimizer(model) {
	lr_ = lr;
	momentum_ = momentum;
	n_params_ = 0;
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		if (layers_[i_layer]->HasWeight()) {
			auto ptr = layers_[i_layer]->GetWeightPointer();
			param_.push_back(ptr);
			grad_.push_back(layers_[i_layer]->GetWeightGradPointer());
			v_.push_back(Zeros(*ptr));
			n_params_++;
		}
		if (layers_[i_layer]->HasBias()) {
			auto ptr = layers_[i_layer]->GetBiasPointer();
			param_.push_back(ptr);
			grad_.push_back(layers_[i_layer]->GetBiasGradPointer());
			v_.push_back(Zeros(*ptr));
			n_params_++;
		}
	}
}


SGD::~SGD() {
}


void SGD::Step() {
	for (int i_param = 0; i_param < n_params_; ++i_param) {
		v_[i_param] = momentum_ * v_[i_param] + *(grad_[i_param]);
		*(param_[i_param]) += lr_ * v_[i_param];
	}
}
