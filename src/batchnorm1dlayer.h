// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Batch normalization layer for 1-D cases.

#ifndef __BATCH_NORM_1D_LAYER_H__
#define __BATCH_NORM_1D_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class BatchNorm1dLayer : public Layer {
public:
	BatchNorm1dLayer(int num_features, float eps = 1.0e-5, float momentum = 0.1);
	virtual ~BatchNorm1dLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
	int num_features_;
	int batch_size_;
	float eps_;
	float momentum_;
	ftensor::Tensor e_input_;
	ftensor::Tensor var_input_;
	ftensor::Tensor h_e_input_;
	ftensor::Tensor h_var_input_;
	ftensor::Tensor r_e_input_;
	ftensor::Tensor r_var_input_;
	ftensor::Tensor r_w_;
	ftensor::Tensor r_b_;
	ftensor::Tensor xi_;
};


#endif  // __BATCH_NORM_1D_LAYER_H__
