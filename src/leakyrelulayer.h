// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: Leaky ReLU layer.

#ifndef __LEAKY_RELU_LAYER_H__
#define __LEAKY_RELU_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class LeakyReluLayer : public Layer {
public:
	LeakyReluLayer(float negative_slope = 0.01);
	virtual ~LeakyReluLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
	float negative_slope_;
};


#endif  // __LEAKY_RELU_LAYER_H__
