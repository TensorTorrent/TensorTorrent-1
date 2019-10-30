// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Learning rate scheduler.

#include "scheduler.h"


Scheduler::Scheduler(Optimizer* optim) {
	optim_ = optim;
	epoch_ = 0;
}


Scheduler::~Scheduler() {
}
