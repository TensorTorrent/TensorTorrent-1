// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Layer pool.

#ifndef __LAYER_POOL_H__
#define __LAYER_POOL_H__


#include <iostream>
#include <vector>

#include "tensorlib.h"
#include "layer.h"
#include "conv2dlayer.h"
#include "convtranspose2dlayer.h"
#include "flattenlayer.h"
#include "relulayer.h"
#include "leakyrelulayer.h"
#include "softmaxlayer.h"
#include "tanhlayer.h"
#include "sigmoidlayer.h"
#include "linearlayer.h"
#include "maxpool2dlayer.h"
#include "batchnorm1dlayer.h"
#include "batchnorm2dlayer.h"
#include "identitylayer.h"
#include "sequential.h"


class LayerPool {
public:
	static LayerPool &GetInstance();
	void Append(Layer* layer);
	void Clear();

private:
	LayerPool();
	~LayerPool();
	LayerPool(const LayerPool &layer_pool);
	const LayerPool &operator=(const LayerPool &layer_pool);

	std::vector<Layer*> layers_;
	int n_layers_;
};


#endif  // __LAYER_POOL_H__
