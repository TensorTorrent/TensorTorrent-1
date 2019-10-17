// Author: Yuning Jiang
// Date: Oct. 17 th, 2019
// Description: MSE loss.

#include "mseloss.h"


using namespace ftensor;


MSELoss::MSELoss()
: Loss() {
}


MSELoss::~MSELoss() {
}


void MSELoss::operator()(const ftensor::Tensor& outputs, const ftensor::Tensor& labels) {
	// Loss
	Tensor goal = 0 * outputs;
	int n_examples = labels.numel();
	Tensor reshaped_labels = Reshape(labels, 1, n_examples, 1, 1);
	for (int i_example = 0; i_example < n_examples; ++i_example) {
		goal(reshaped_labels(i_example), i_example) = 1;
	}
	Tensor predictions;
	Max(outputs, 0, &predictions);

	correct_ = Sum(Where(Round(reshaped_labels) == Round(predictions), 1, 0));

	Tensor mse = Pow(goal - outputs, 2);
	loss_ = Sum(mse);

	// Gradient
	grad_ = goal - outputs;
}
