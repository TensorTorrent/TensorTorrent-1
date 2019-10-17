// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: Tanh layer.

#ifndef __TANH_LAYER_H__
#define __TANH_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class TanhLayer : public Layer {
public:
	TanhLayer();
	virtual ~TanhLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
};


#endif  // __TANH_LAYER_H__
