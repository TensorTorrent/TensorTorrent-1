// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: Sigmoid layer.

#ifndef __SIGMOID_LAYER_H__
#define __SIGMOID_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class SigmoidLayer : public Layer {
public:
	SigmoidLayer();
	virtual ~SigmoidLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
};


#endif  // __SIGMOID_LAYER_H__
