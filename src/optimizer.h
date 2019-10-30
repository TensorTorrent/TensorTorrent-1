// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Optimizer.

#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"
#include "model.h"


class Optimizer {
public:
	Optimizer(Model& model);
	virtual ~Optimizer();
	void ZeroGrad();
	virtual void Step() = 0;

	float GetLearningRate() const {return lr_;}
	void SetLearningRate(float lr) {lr_ = lr;}

protected:
	float lr_;
	std::vector<Layer*> layers_;
	int n_layers_;
};


#endif  // __OPTIMIZER_H__
