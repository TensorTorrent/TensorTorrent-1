// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Crossentropy loss.

#ifndef __CROSS_ENTROPY_LOSS_H__
#define __CROSS_ENTROPY_LOSS_H__


#include <iostream>

#include "tensorlib.h"
#include "loss.h"


class CrossEntropyLoss : public Loss {
public:
	CrossEntropyLoss();
	virtual ~CrossEntropyLoss();

	void operator()(const ftensor::Tensor& outputs, const ftensor::Tensor& labels);

protected:
};


#endif  // __CROSS_ENTROPY_LOSS_H__
