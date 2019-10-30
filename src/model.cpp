// Author: Yuning Jiang
// Date: Oct. 29 th, 2019
// Description: Save and load models.

#include "model.h"


using std::cerr;
using std::endl;


Model::Model() {
	n_layers_ = 0;
}


Model::Model(Layer* layer) {
	n_layers_ = 1;
	layer->SetAsFirstLayer();
	layers_.push_back(layer);
}


Model::Model(std::initializer_list<Layer*> layers) {
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


Model::~Model() {
}


ftensor::Tensor Model::Forward(const ftensor::Tensor& input) {
	layers_[0]->SetInput(input);
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		layers_[i_layer]->Forward();
	}
	return layers_[n_layers_ - 1]->GetOutput();
}


ftensor::Tensor Model::Backward(const ftensor::Tensor& gradient) {
	layers_[n_layers_ - 1]->SetGradInput(gradient);
	for (int i_layer = n_layers_ - 1; i_layer >= 0; --i_layer) {
		layers_[i_layer]->Backward();
	}
	return layers_[0]->GetGradOutput();
}


void Model::Train(bool mode) {
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		layers_[i_layer]->Train(mode);
	}
}


void Model::Eval() {
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		layers_[i_layer]->Eval();
	}
}


void Model::Append(Layer* layer) {
	if (0 == n_layers_) {
		layer->SetAsFirstLayer();
	}
	else {
		layer->SetPreviousLayer(layers_[n_layers_ - 1]);
	}
	layers_.push_back(layer);
	n_layers_++;
}


void Model::Clear() {
	layers_.clear();
	n_layers_ = 0;
}


void Model::ExportTo(std::ofstream& output_file) {
	int32_t end_of_model = END_OF_MODEL;
	for (int i_layer = 0; i_layer < n_layers_; ++i_layer) {
		layers_[i_layer]->ExportTo(output_file);
	}
	output_file.write((char*)&end_of_model, sizeof(int32_t));
}


void Model::ImportFrom(std::ifstream& input_file) {
	int32_t layer_type_id = 0;
	layers_.clear();
	n_layers_ = 0;
	while (true) {
		input_file.read((char *)&layer_type_id, sizeof(int32_t));
		if (END_OF_MODEL == layer_type_id) {
			break;
		}
		else {
			Layer* new_layer;
			switch (layer_type_id) {
				case 0:
				cerr << "Error: Layer class is an abstract class thus cannot be instantiated." << endl;
				exit(1);
				break;
				case 1:
				new_layer = new IdentityLayer;
				break;
				case 2:
				new_layer = new LinearLayer(1, 1);
				break;
				case 3:
				new_layer = new ReluLayer;
				break;
				case 4:
				new_layer = new SoftmaxLayer;
				break;
				case 5:
				new_layer = new Conv2dLayer(1, 1, 1);
				break;
				case 6:
				new_layer = new ConvTranspose2dLayer(1, 1, 1);
				break;
				case 7:
				new_layer = new BatchNorm1dLayer(1);
				break;
				case 8:
				new_layer = new BatchNorm2dLayer(1);
				break;
				case 9:
				new_layer = new TanhLayer;
				break;
				case 10:
				new_layer = new SigmoidLayer;
				break;
				case 11:
				new_layer = new MaxPool2dLayer;
				break;
				case 12:
				new_layer = new LeakyReluLayer;
				break;
				case 13:
				new_layer = new FlattenLayer;
				break;
				default:
				cerr << "Error: Unsupported layer type." << endl;
				exit(1);
			}
			new_layer->ImportFrom(input_file);
			LayerPool::GetInstance().Append(new_layer);
			Append(new_layer);
		}
	}
}


void Save(Model* model, std::string file_name) {
	std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
	if(!output_file) {
		cerr << "Error: File creation failed." << endl;
		exit(1);
	}
	else {
		model->ExportTo(output_file);
	}
	output_file.close();
}


void Save(Layer* layer, std::string file_name) {
	Model model(layer);
	Save(&model, file_name);
}


Model Load(std::string file_name) {
	Model model;
	std::ifstream input_file(file_name, std::ios::in | std::ios::binary);
	if(!input_file) {
		cerr << "Error: File does NOT exist." << endl;
		exit(1);
	}
	else {
		model.ImportFrom(input_file);
	}
	input_file.close();
	return model;
}
