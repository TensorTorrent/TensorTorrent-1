// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Data loader.

#ifndef __DATALOADER_H__
#define __DATALOADER_H__


#include <iostream>
#include <vector>

#include "tensorlib.h"
#include "dataset.h"


std::vector<Dataset> DataLoader(Dataset dataset, int batch_size, bool shuffle = false);


#endif  // __DATALOADER_H__
