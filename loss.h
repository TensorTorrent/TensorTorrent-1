// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Loss function.

#ifndef __LOSS_H__
#define __LOSS_H__


#include <iostream>

#include "tensorlib.h"


class Loss {
public:
	Loss();
	virtual ~Loss();

	virtual void operator()(const ftensor::Tensor& outputs, const ftensor::Tensor& labels) = 0;
	const ftensor::Tensor& GetLoss() {return loss_;}
	const ftensor::Tensor& GetGrad() {return grad_;}
	const ftensor::Tensor& GetCorrect() {return correct_;}

protected:
	ftensor::Tensor loss_;
	ftensor::Tensor grad_;
	ftensor::Tensor correct_;
};


#endif  // __LOSS_H__
