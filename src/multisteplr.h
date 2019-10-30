// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Set the learning rate of each parameter group to the initial lr decayed by gamma once the number of epoch reaches one of the milestones.

#ifndef __MULTI_STEP_LR_H__
#define __MULTI_STEP_LR_H__


#include <iostream>
#include <vector>

#include "tensorlib.h"
#include "optimizer.h"
#include "scheduler.h"


class MultiStepLR : public Scheduler {
public:
	MultiStepLR(Optimizer* optim, const std::vector<int>& milestones, float gamma = 0.1);
	virtual ~MultiStepLR();

	void Step();

protected:
	std::vector<int> milestones_;
	unsigned int milestones_index_;
	float gamma_;
};


#endif  // __MULTI_STEP_LR_H__
