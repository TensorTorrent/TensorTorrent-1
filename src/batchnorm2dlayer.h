// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Batch normalization layer for 2-D cases.

#ifndef __BATCH_NORM_2D_LAYER_H__
#define __BATCH_NORM_2D_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class BatchNorm2dLayer : public Layer {
public:
	BatchNorm2dLayer(int num_features, float eps = 1.0e-5);
	virtual ~BatchNorm2dLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
	int shape_[4];
	int num_features_;
	int batch_size_;
	float eps_;
	ftensor::Tensor e_input_;
	ftensor::Tensor var_input_;
	ftensor::Tensor r_e_input_;
	ftensor::Tensor r_var_input_;
	ftensor::Tensor r_w_;
	ftensor::Tensor r_b_;
	ftensor::Tensor xi_;
};


#endif  // __BATCH_NORM_2D_LAYER_H__
