// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Flatten layer.

#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class FlattenLayer : public Layer {
public:
	FlattenLayer(int start_dim = 0, int end_dim = 2);
	virtual ~FlattenLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
	int start_dim_;
	int end_dim_;
	int shape_[4];
};


#endif  // __FLATTEN_LAYER_H__
