// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs.

#include "steplr.h"


StepLR::StepLR(Optimizer* optim, int step_size, float gamma)
: Scheduler(optim) {
	step_size_ = step_size;
	gamma_ = gamma;
}


StepLR::~StepLR() {
}


void StepLR::Step() {
	epoch_++;
	if (0 == (epoch_ + 1) % step_size_) {
		optim_->SetLearningRate(gamma_ * optim_->GetLearningRate());
	}
}
