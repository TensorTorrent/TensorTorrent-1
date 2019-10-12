// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Dataset.

#ifndef __DATASET_H__
#define __DATASET_H__


#include <iostream>

#include "tensorlib.h"


class Dataset {
public:
	Dataset();
	Dataset(const ftensor::Tensor& new_data, const ftensor::Tensor& new_labels);
	virtual ~Dataset();

	void Import(const ftensor::Tensor& new_data, const ftensor::Tensor& new_labels);
	int Size() {return n_examples_;}
	ftensor::Tensor& data() {return data_;}
	ftensor::Tensor& labels() {return labels_;}

protected:
	ftensor::Tensor data_;
	ftensor::Tensor labels_;
	int n_examples_;
};


#endif  // __DATASET_H__
