// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Crossentropy loss.

#include "crossentropyloss.h"


using namespace ftensor;


const float LIMIT = 1e-6;


CrossEntropyLoss::CrossEntropyLoss()
: Loss() {
}


CrossEntropyLoss::~CrossEntropyLoss() {
}


void CrossEntropyLoss::operator()(const ftensor::Tensor& outputs, const ftensor::Tensor& labels) {
	// Loss
	Tensor goal = Zeros(outputs);
	int n_examples = labels.numel();
	Tensor reshaped_labels = Reshape(labels, 1, n_examples, 1, 1);
	for (int i_example = 0; i_example < n_examples; ++i_example) {
		goal(reshaped_labels(i_example) - 1, i_example) = 1;
	}
	Tensor predictions;
	Max(outputs, 0, &predictions);

	correct_ = Sum(Where(Round(reshaped_labels) == Round(predictions), 1, 0));
	Tensor yn0 = Where(outputs > 1 - LIMIT, 1 - LIMIT, outputs);
	Tensor yn = Where(yn0 < LIMIT, LIMIT, yn0);

	Tensor crossentropy = -Mul(goal, Log(yn)) - Mul(1.0 - goal, Log(1.0 - yn));
	loss_ = Sum(crossentropy);

	// Gradient
	grad_ = goal - outputs;
}
