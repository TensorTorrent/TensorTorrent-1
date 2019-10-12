// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Adam optimizer.

#ifndef __ADAM_H__
#define __ADAM_H__


#include <iostream>

#include "tensorlib.h"
#include "layer.h"
#include "sequential.h"
#include "optimizer.h"


class Adam : public Optimizer {
public:
	Adam(Sequential& model, float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-08);
	virtual ~Adam();
	void Step();

protected:
	float lr_;
	float beta1_;
	float beta2_;
	float eps_;
};


#endif  // __ADAM_H__
