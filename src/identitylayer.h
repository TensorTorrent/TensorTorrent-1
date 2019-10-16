// Author: Yuning Jiang
// Date: Oct. 16 th, 2019
// Description: Identity layer.

#ifndef __IDENTITY_LAYER_H__
#define __IDENTITY_LAYER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"


class IdentityLayer : public Layer {
public:
	IdentityLayer();
	virtual ~IdentityLayer();

	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

protected:
};


#endif  // __IDENTITY_LAYER_H__
