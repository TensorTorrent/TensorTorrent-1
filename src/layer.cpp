// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Layers.

#include "layer.h"


using namespace ftensor;
using std::cerr;
using std::endl;


Layer::Layer() {
	layer_type_id_ = 0;
	is_first_layer_ = true;
	previous_layer_ = nullptr;
	has_weight_ = false;
	has_bias_ = false;
	training_mode_ = false;
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
	if (has_weight_) {
		dw_.Zeros();
	}
	if (has_bias_) {
		db_.Zeros();
	}
}


void Layer::SetWeight(const ftensor::Tensor& w) {
	if (has_weight_) {
		if (Match(w, w_)) {
			w_ = w;
		}
		else {
			cerr << "Error: Dimension mismatched in weight assignment." << endl;
			exit(1);
		}
	}
	else {
		cerr << "Error: The layer has no weight." << endl;
		exit(1);
	}
}


void Layer::SetBias(const ftensor::Tensor& b) {
	if (has_bias_) {
		if (Match(b, b_)) {
			b_ = b;
		}
		else {
			cerr << "Error: Dimension mismatched in weight assignment." << endl;
			exit(1);
		}
	}
	else {
		cerr << "Error: The layer has no bias." << endl;
		exit(1);
	}
}


void Layer::ExportTo(std::ofstream& output_file) {
	int32_t end_of_layer = END_OF_LAYER;
	output_file.write((char*)&layer_type_id_, sizeof(int32_t));
	output_file.write((char*)&end_of_layer, sizeof(int32_t));
}


void Layer::ImportFrom(std::ifstream& input_file) {
	int32_t end_of_layer = 0;
	input_file.read((char *)&end_of_layer, sizeof(int32_t));
	if (END_OF_LAYER != end_of_layer) {
		cerr << "Error: Invalid model format." << endl;
		exit(1);
	}
}
