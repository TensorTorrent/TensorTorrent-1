// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Adam optimizer.

#ifndef __ADAM_H__
#define __ADAM_H__


#include <iostream>
#include <vector>

#include "tensorlib.h"
#include "layer.h"
#include "model.h"
#include "optimizer.h"


class Adam : public Optimizer {
public:
	Adam(Model& model, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-08);
	virtual ~Adam();
	void Step();

protected:
	float beta1_;
	float beta2_;
	float eps_;
	std::vector<ftensor::Tensor*> param_;
	std::vector<ftensor::Tensor*> grad_;
	std::vector<ftensor::Tensor> m_;
	std::vector<ftensor::Tensor> v_;
	int t_;
	int n_params_;
};


#endif  // __ADAM_H__
