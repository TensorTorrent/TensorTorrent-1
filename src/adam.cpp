// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Adam optimizer.

#include "adam.h"


using namespace ftensor;


Adam::Adam(Model& model, float lr, float beta1, float beta2, float eps)
: Optimizer(model) {
	lr_ = lr;
	beta1_ = beta1;
	beta2_ = beta2;
	eps_ = eps;
	t_ = 0;
	n_params_ = 0;
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		if (layers_[i_layer]->HasWeight()) {
			auto ptr = layers_[i_layer]->GetWeightPointer();
			param_.push_back(ptr);
			grad_.push_back(layers_[i_layer]->GetWeightGradPointer());
			m_.push_back(Zeros(*ptr));
			v_.push_back(Zeros(*ptr));
			n_params_++;
		}
		if (layers_[i_layer]->HasBias()) {
			auto ptr = layers_[i_layer]->GetBiasPointer();
			param_.push_back(ptr);
			grad_.push_back(layers_[i_layer]->GetBiasGradPointer());
			m_.push_back(Zeros(*ptr));
			v_.push_back(Zeros(*ptr));
			n_params_++;
		}
	}
}


Adam::~Adam() {
}


void Adam::Step() {
	t_++;
	for (int i_param = 0; i_param < n_params_; ++i_param) {
		m_[i_param] = beta1_ * m_[i_param] + (1.0 - beta1_) * *(grad_[i_param]);
		v_[i_param] = beta2_ * v_[i_param] + (1.0 - beta2_) * Pow(*(grad_[i_param]), 2);
		Tensor hm1 = m_[i_param] / (1.0 - pow(beta1_, t_));
		Tensor hv1 = v_[i_param] / (1.0 - pow(beta2_, t_));
		*(param_[i_param]) += lr_ * Div(hm1, (Sqrt(hv1) + eps_));
	}
}
