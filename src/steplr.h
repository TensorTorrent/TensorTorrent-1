// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs.

#ifndef __STEP_LR_H__
#define __STEP_LR_H__


#include <iostream>

#include "tensorlib.h"
#include "optimizer.h"
#include "scheduler.h"


class StepLR : public Scheduler {
public:
	StepLR(Optimizer* optim, int step_size, float gamma = 0.1);
	virtual ~StepLR();

	void Step();

protected:
	int step_size_;
	float gamma_;
};


#endif  // __STEP_LR_H__
