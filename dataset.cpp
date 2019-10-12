// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Dataset.

#include "dataset.h"


using namespace ftensor;
using std::cerr;
using std::endl;


Dataset::Dataset() {
}


Dataset::Dataset(const ftensor::Tensor& new_data, const ftensor::Tensor& new_labels) {
	Import(new_data, new_labels);
}


Dataset::~Dataset() {
}


void Dataset::Import(const Tensor& new_data, const Tensor& new_labels) {
	if (new_data.gros() == new_labels.gros()) {
		data_ = new_data;
		labels_ = new_labels;
		n_examples_ = new_labels.gros();
	}
	else {
		cerr << "Error: The 3rd dimension of data and labels should match." << endl;
		exit(1);
	}
}
