// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Max pooling 2D layer.

#ifndef __MAX_POOL_2D_LAYER_H__
#define __MAX_POOL_2D_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class MaxPool2dLayer : public Layer {
public:
	MaxPool2dLayer(int kernel_size = 2);
	virtual ~MaxPool2dLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
	int kernel_size_;
	ftensor::Tensor mask_;
};


#endif  // __MAX_POOL_2D_LAYER_H__
