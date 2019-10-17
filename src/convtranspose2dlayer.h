// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Transposed convolution layer.

#ifndef __CONVTRANSPOSE2D_LAYER_H__
#define __CONVTRANSPOSE2D_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class ConvTranspose2dLayer : public Layer {
public:
	ConvTranspose2dLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true);
	virtual ~ConvTranspose2dLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
	int in_channels_;
	int out_channels_;
	int kernel_size_;
	int stride_;
	int padding_;
	float init_range_;
};


#endif  // __CONVTRANSPOSE2D_LAYER_H__
