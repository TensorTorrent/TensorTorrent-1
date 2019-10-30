// Author: Yuning Jiang
// Date: Oct. 29 th, 2019
// Description: Save and load models.

#ifndef __MODEL_H__
#define __MODEL_H__


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <initializer_list>

#include "tensorlib.h"
#include "layer.h"
#include "conv2dlayer.h"
#include "convtranspose2dlayer.h"
#include "flattenlayer.h"
#include "relulayer.h"
#include "leakyrelulayer.h"
#include "softmaxlayer.h"
#include "tanhlayer.h"
#include "sigmoidlayer.h"
#include "linearlayer.h"
#include "maxpool2dlayer.h"
#include "batchnorm1dlayer.h"
#include "batchnorm2dlayer.h"
#include "identitylayer.h"
#include "sequential.h"
#include "layerpool.h"


class Model {
public:
	Model();
	Model(Layer* layer);
	Model(std::initializer_list<Layer*> layers);
	Model(const Model& model) {n_layers_ = model.GetLayerNum(); layers_ = model.GetLayers();}
	virtual ~Model();

	ftensor::Tensor operator()(const ftensor::Tensor& input) {return Forward(input);}
	void operator=(const Model& model) {n_layers_ = model.GetLayerNum(); layers_ = model.GetLayers();}
	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

	std::vector<Layer*> GetLayers() const {return layers_;}
	int GetLayerNum() const {return n_layers_;}

	void Train(bool mode = true);
	void Eval();

	void Append(Layer* layer);
	void Clear();
	void ExportTo(std::ofstream& output_file);
	void ImportFrom(std::ifstream& input_file);
	
protected:
	std::vector<Layer*> layers_;
	int n_layers_;
};


void Save(Model* model, std::string file_name);
void Save(Layer* layer, std::string file_name);
Model Load(std::string file_name);


#endif  // __MODEL_H__
