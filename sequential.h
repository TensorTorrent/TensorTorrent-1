// Author: Yuning Jiang
// Date: Oct. 11 th, 2019
// Description: Sequential container

#ifndef __SEQUENTIAL_H__
#define __SEQUENTIAL_H__

#include <iostream>
#include <initializer_list>
#include <ctime>
#include "tensorlib.h"
#include "layer.h"
#include "conv2dlayer.h"
#include "convtranspose2dlayer.h"
#include "flattenlayer.h"
#include "relulayer.h"
#include "softmaxlayer.h"
#include "linearlayer.h"
#include "maxpool2dlayer.h"


class Sequential {
public:
	Sequential(std::initializer_list<Layer*> layers);
	virtual ~Sequential();

	ftensor::Tensor operator()(const ftensor::Tensor& input) {return Forward(input);}
	ftensor::Tensor Forward(const ftensor::Tensor& input);
	ftensor::Tensor Backward(const ftensor::Tensor& gradient);

private:
	std::vector<Layer*> layers_;
	int n_layers_;
};


#endif // __SEQUENTIAL_H__
