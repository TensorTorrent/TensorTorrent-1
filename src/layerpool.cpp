// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Layer pool.

#include "layerpool.h"


LayerPool &LayerPool::GetInstance() {
	static LayerPool layer_pool;
	return layer_pool;
}


void LayerPool::Append(Layer* layer) {
	layers_.push_back(layer);
	n_layers_++;
}


void LayerPool::Clear() {
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		if (nullptr != layers_[i_layer]) {
			delete layers_[i_layer];
			layers_[i_layer] = nullptr;
		}
	}
	layers_.clear();
	n_layers_ = 0;
}


LayerPool::LayerPool() {
	n_layers_ = 0;
}


LayerPool::~LayerPool() {
	Clear();
}
