// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Linear layer.

#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class LinearLayer : public Layer {
public:
	LinearLayer(int in_features, int out_features, bool bias = true);
	virtual ~LinearLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input_image);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);
	void ZeroGrad();

protected:
	int in_features_;
	int out_features_;
	float init_range_;
};


#endif  // __LINEAR_LAYER_H__
