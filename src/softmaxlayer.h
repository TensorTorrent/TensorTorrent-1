// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Softmax layer.

#ifndef __SOFTMAX_LAYER_H__
#define __SOFTMAX_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class SoftmaxLayer : public Layer {
public:
	SoftmaxLayer(int dim = 0);
	virtual ~SoftmaxLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
	int dim_;
};


#endif  // __SOFTMAX_LAYER_H__
