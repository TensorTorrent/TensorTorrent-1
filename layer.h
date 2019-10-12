// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Layers.

#ifndef __LAYER_H__
#define __LAYER_H__


#include <iostream>

#include "tensorlib.h"


class Layer {
public:
	Layer();
	virtual ~Layer();
	
	ftensor::Tensor operator()(const ftensor::Tensor& input) {return Forward(input);}
	virtual ftensor::Tensor Forward(const ftensor::Tensor& input) = 0;
	virtual ftensor::Tensor Backward(const ftensor::Tensor& input) = 0;
	void Forward();
	void Backward();
	virtual void ZeroGrad();

	void SetAsFirstLayer() {is_first_layer_ = true; previous_layer_ = nullptr;}
	void SetPreviousLayer(Layer* previous_layer) {previous_layer_ = previous_layer; is_first_layer_ = false;}
	void SetInput(const ftensor::Tensor& input) {input_ = input;}
	const ftensor::Tensor& GetOutput() {return output_;}
	void SetGradInput(const ftensor::Tensor& grad_input) {grad_input_ = grad_input;}
	const ftensor::Tensor& GetGradOutput() {return grad_output_;}

protected:
	bool is_first_layer_;
	Layer* previous_layer_;
	ftensor::Tensor input_;
	ftensor::Tensor output_;
	ftensor::Tensor grad_input_;
	ftensor::Tensor grad_output_;
};


#endif  // __LAYER_H__
