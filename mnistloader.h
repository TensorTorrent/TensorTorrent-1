// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: MNIST loader.

#ifndef __MNIST_LOADER_H__
#define __MNIST_LOADER_H__


#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "tensorlib.h"

using std::vector;


inline void EndianConvert(int &data);
void LoadMnistLabels(std::string label_file_name, std::vector<float>&labels);
void LoadMnistImages(std::string image_file_name, std::vector<std::vector<float> >&images);

void LoadDatabase(const std::string& path, ftensor::Tensor& training_images, ftensor::Tensor& training_labels, \
	ftensor::Tensor& testing_images, ftensor::Tensor& testing_labels);

#endif  // __MNIST_LOADER_H__
