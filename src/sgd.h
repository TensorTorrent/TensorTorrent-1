// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: SGD optimizer.

#ifndef __SGD_H__
#define __SGD_H__


#include <iostream>
#include <vector>

#include "tensorlib.h"
#include "layer.h"
#include "model.h"
#include "optimizer.h"


class SGD : public Optimizer {
public:
	SGD(Model& model, float lr = 0.1, float momentum = 0.9);
	virtual ~SGD();
	void Step();

protected:
	float momentum_;
	std::vector<ftensor::Tensor*> param_;
	std::vector<ftensor::Tensor*> grad_;
	std::vector<ftensor::Tensor> v_;
	int n_params_;
};


#endif  // __SGD_H__
