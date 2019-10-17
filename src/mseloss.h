// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: MSE loss.

#ifndef __MSE_LOSS_H__
#define __MSE_LOSS_H__


#include <iostream>

#include "tensorlib.h"
#include "loss.h"


class MSELoss : public Loss {
public:
	MSELoss();
	virtual ~MSELoss();

	void operator()(const ftensor::Tensor& outputs, const ftensor::Tensor& labels);

protected:
};


#endif  // __MSE_LOSS_H__
