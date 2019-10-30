// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Optimizer.

#include "optimizer.h"


using namespace ftensor;


Optimizer::Optimizer(Model& model) {
	lr_ = 0.0;
	layers_ = model.GetLayers();
	n_layers_ = model.GetLayerNum();
}


Optimizer::~Optimizer() {
}


void Optimizer::ZeroGrad() {
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		layers_[i_layer]->ZeroGrad();
	}
}
