// Author: Yuning Jiang
// Date: Oct. 12 th, 2019
// Description: Data loader.

#include "dataloader.h"


using namespace ftensor;
using std::vector;
using std::min;


vector<Dataset> DataLoader(Dataset dataset, int batch_size, bool shuffle) {
	vector<Dataset> v;
	Tensor data;
	Tensor labels;
	if (shuffle) {
		vector<int> rand_index = RandomIndex(dataset.Size());
		data = Rearrange(dataset.data(), rand_index, 3);
		labels = Rearrange(dataset.labels(), rand_index, 3);
	}
	else {
		data = dataset.data();
		labels = dataset.labels();
	}
	int n_examples = labels.numel();
	int n_batches = int(ceil(1.0 * n_examples / batch_size));
	for (int i_batch = 0; i_batch < n_batches; ++i_batch) {
		int start_example = batch_size * i_batch;
		int end_example = min(batch_size * (i_batch + 1), n_examples);
		Dataset subset(data.S(-1, -1, -1, -1, -1, -1, start_example, end_example), labels.S(-1, -1, -1, -1, -1, -1, start_example, end_example));
		v.push_back(subset);
	}
	return v;
}
