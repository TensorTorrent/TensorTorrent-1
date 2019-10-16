// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: ReLU layer.

#ifndef __RELU_LAYER_H__
#define __RELU_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class ReluLayer : public Layer {
public:
	ReluLayer();
	virtual ~ReluLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
};


#endif  // __RELU_LAYER_H__
