// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer implemented by GOKU chips.

#ifndef __CONV2D_GOKU_H__
#define __CONV2D_GOKU_H__


#include <iostream>

#include "tensorlib.h"
#include "conv2dlayer.h"
#include "goku.h"


class Conv2dGoku : public Conv2dLayer {
public:
	Conv2dGoku(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
	virtual ~Conv2dGoku();

	itensor32::Tensor WorkOutKernelMap(const itensor32::Tensor& kernel);
	itensor32::Tensor WorkOutKernelIds(int rpixel, int cpixel, int orows, int ocols, int stride, int padding);

	void to(Goku* goku_chip, bool sim = false);

	itensor32::Tensor operator()(const itensor32::Tensor& input_image);

private:
	Goku* goku_chip_;
	itensor32::Tensor kernel_ids_;
	itensor32::Tensor kernel_value_map_;
};


#endif  // __CONV2D_GOKU_H__
