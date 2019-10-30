// Author: Yuning Jiang
// Date: Oct. 30 th, 2019
// Description: Set the learning rate of each parameter group to the initial lr decayed by gamma once the number of epoch reaches one of the milestones.

#include "multisteplr.h"


MultiStepLR::MultiStepLR(Optimizer* optim, const std::vector<int>& milestones, float gamma)
: Scheduler(optim) {
	milestones_ = milestones;
	milestones_index_ = 0;
	gamma_ = gamma;
}


MultiStepLR::~MultiStepLR() {
}


void MultiStepLR::Step() {
	epoch_++;
	if (milestones_index_ < milestones_.size()) {
		if (epoch_ + 1 == milestones_[milestones_index_]) {
			optim_->SetLearningRate(gamma_ * optim_->GetLearningRate());
			milestones_index_++;
		}
	}
}
