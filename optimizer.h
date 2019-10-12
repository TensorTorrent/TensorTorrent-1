// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Optimizer.

#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"
#include "sequential.h"


class Optimizer {
public:
	Optimizer(Sequential& model);
	virtual ~Optimizer();
	virtual void ZeroGrad();
	virtual void Step() = 0;

protected:
	std::vector<Layer*> layers_;
	int n_layers_;
};


#endif  // __OPTIMIZER_H__
