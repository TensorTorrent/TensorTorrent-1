// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Learning rate scheduler.

#ifndef __SCHEDULE_H__
#define __SCHEDULE_H__


#include <iostream>

#include "tensorlib.h"
#include "optimizer.h"


class Scheduler {
public:
	Scheduler(Optimizer* optim);
	virtual ~Scheduler();

	virtual void Step() = 0;

protected:
	Optimizer* optim_;
	int epoch_;
};


#endif  // __SCHEDULE_H__
