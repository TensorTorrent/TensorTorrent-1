#include <iostream>
#include <ctime>
#include "tensortorrent.h"


using namespace ftensor;
using namespace std;


const int kBatchSize = 100;
const int kNumEpochs = 1;


int main() {
	clock_t start, end;
	
	// Load the MNIST database
	Dataset trainset, testset;
	MnistLoader("/opt/Datasets/MNIST", trainset, testset);
	trainset.data() /= 255.0;
	testset.data() /= 255.0;
	vector<Dataset> train_loader = DataLoader(trainset, kBatchSize, true);
	vector<Dataset> test_loader = DataLoader(testset, kBatchSize, false);
	
	// Define layers
	Conv2dLayer c1(1, 6, 5, 1, 2, false);
	BatchNorm2dLayer b2(6);
	ReluLayer r3;
	MaxPool2dLayer m4;
	Conv2dLayer c5(6, 16, 5, 1, 0, false);
	BatchNorm2dLayer b6(16);
	ReluLayer r7;
	MaxPool2dLayer m8;
	FlattenLayer f9;
	LinearLayer l10(400, 120, false);
	BatchNorm1dLayer b11(120);
	ReluLayer r12;
	LinearLayer l13(120, 84, false);
	BatchNorm1dLayer b14(84);
	ReluLayer r15;
	LinearLayer l16(84, 10, false);
	BatchNorm1dLayer b17(10);
	SoftmaxLayer s18;
	
	// Construct the model
	Sequential net({&c1, &b2, &r3, &m4, &c5, &b6, &r7, &m8, &f9, &l10, &b11, &r12, &l13, &b14, &r15, &l16, &b17, &s18});

	// Optimizer and loss function
	Adam optimizer(net);
	CrossEntropyLoss criterion;

	for (int i_epoch = 0; i_epoch < kNumEpochs + 1; ++i_epoch) {
		start = clock();

		if (0 != i_epoch) {
			// Train
			float running_loss = 0.0;
			int train_correct = 0;
			int train_total = 0;
			int batch_idx = 0;
			for (auto train_iter = train_loader.begin(); train_iter != train_loader.end(); ++train_iter) {
				batch_idx++;
				auto train_labels = train_iter->labels();
				optimizer.ZeroGrad();
				auto outputs = net(train_iter->data());
				criterion(outputs, train_labels);
				auto loss = criterion.GetLoss();
				net.Backward(criterion.GetGrad());
				optimizer.Step();
				running_loss += loss.Item();
				train_total += train_labels.Size(3);
				auto correct = criterion.GetCorrect();
				train_correct += correct.Item();
				if (0 == batch_idx % 60) {
					running_loss /= train_total;
					float train_accuracy = 100.0 * train_correct / train_total;
					cout << "[Train]  Epoch:" << i_epoch << "\t\t-\t\tLoss: " << fixed << setprecision(4) << running_loss << "\t\t-\t\tAccuracy: ";
					cout << setprecision(2) << train_accuracy << "%" << endl << flush;
					running_loss = 0.0;
					train_correct = 0;
					train_total = 0;
				}
			}
		}

		// Test
		float test_loss = 0.0;
		int test_correct = 0;
		int test_total = 0;
		for (auto test_iter = test_loader.begin(); test_iter != test_loader.end(); ++test_iter) {
			auto test_images = test_iter->data();
			auto test_labels = test_iter->labels();
			auto outputs = net(test_images);
			criterion(outputs, test_labels);
            auto loss = criterion.GetLoss();
            test_loss += loss.Item();
            test_total += test_labels.Size(3);
            auto correct = criterion.GetCorrect();
            test_correct += correct.Item();
		}
		test_loss /= test_total;
		float test_accuracy = 100.0 * test_correct / test_total;
		cout << "\n[Test]   Epoch:" << i_epoch << "\t\t-\t\tLoss: " << fixed << setprecision(4) << test_loss << "\t\t-\t\tAccuracy: ";
		cout << setprecision(2) << test_accuracy << "%" << endl << flush;

		end = clock();
		cout << "Time elapsed: " << fixed << setprecision(9) << ((double)(end - start) / CLOCKS_PER_SEC) << " s." << endl << endl;
	}
	
	return 0;
}
