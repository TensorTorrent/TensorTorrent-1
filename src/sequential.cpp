// Author: Yuning Jiang
// Date: Jun. 2 nd, 2019
// Description: Fully-connected neural networks

#include "sequential.h"


using namespace ftensor;
using namespace std;


Sequential::Sequential(std::initializer_list<Layer*> layers) {
	n_layers_ = 0;
	for (auto &a : layers) {
		if (0 == n_layers_) {
			a->SetAsFirstLayer();
		}
		else {
			a->SetPreviousLayer(layers_[n_layers_ - 1]);
		}
		layers_.push_back(a);
		n_layers_++;
	}
}


Sequential::~Sequential() {
}


Tensor Sequential::Forward(const Tensor& input) {
	layers_[0]->SetInput(input);
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		layers_[i_layer]->Forward();
	}
	return layers_[n_layers_ - 1]->GetOutput();
}


Tensor Sequential::Backward(const Tensor& gradient) {
	layers_[n_layers_ - 1]->SetGradInput(gradient);
	for (int i_layer = n_layers_ - 1; i_layer >= 0; --i_layer) {
		layers_[i_layer]->Backward();
	}
	return layers_[0]->GetGradOutput();
}
