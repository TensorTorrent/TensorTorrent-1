// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Layers.

#ifndef __LAYER_H__
#define __LAYER_H__


#include <iostream>
#include <fstream>

#include "tensorlib.h"

#define END_OF_LAYER 0x4BC27A50
#define END_OF_CONTAINER 0x1C82F074
#define END_OF_MODEL 0x7823E1D6


class Layer {
public:
	Layer();
	virtual ~Layer();
	
	ftensor::Tensor operator()(const ftensor::Tensor& input) {return Forward(input);}
	virtual ftensor::Tensor Forward(const ftensor::Tensor& input) = 0;
	virtual ftensor::Tensor Backward(const ftensor::Tensor& input) = 0;
	void Forward();
	void Backward();
	void ZeroGrad();

	void SetAsFirstLayer() {is_first_layer_ = true; previous_layer_ = nullptr;}
	void SetPreviousLayer(Layer* previous_layer) {previous_layer_ = previous_layer; is_first_layer_ = false;}
	void SetInput(const ftensor::Tensor& input) {input_ = input;}
	const ftensor::Tensor& GetOutput() {return output_;}
	void SetGradInput(const ftensor::Tensor& grad_input) {grad_input_ = grad_input;}
	const ftensor::Tensor& GetGradOutput() {return grad_output_;}

	bool HasWeight() {return has_weight_;}
	bool HasBias() {return has_bias_;}
	ftensor::Tensor* GetWeightPointer() {return &w_;}
	ftensor::Tensor* GetWeightGradPointer() {return &dw_;}
	ftensor::Tensor* GetBiasPointer() {return &b_;}
	ftensor::Tensor* GetBiasGradPointer() {return &db_;}

	void SetWeight(const ftensor::Tensor& w);
	const ftensor::Tensor& GetWeight() {return w_;}
	void SetBias(const ftensor::Tensor& b);
	const ftensor::Tensor& GetBias() {return b_;}

	void Train(bool mode = true) {training_mode_ = mode;}
	void Eval() {training_mode_ = false;}

	virtual void ExportTo(std::ofstream& output_file);
	virtual void ImportFrom(std::ifstream& input_file);

protected:
	int32_t layer_type_id_;
	bool is_first_layer_;
	bool has_weight_;
	bool has_bias_;
	bool training_mode_;
	Layer* previous_layer_;
	ftensor::Tensor input_;
	ftensor::Tensor output_;
	ftensor::Tensor grad_input_;
	ftensor::Tensor grad_output_;
	ftensor::Tensor w_;
	ftensor::Tensor dw_;
	ftensor::Tensor b_;
	ftensor::Tensor db_;
};


#endif  // __LAYER_H__
