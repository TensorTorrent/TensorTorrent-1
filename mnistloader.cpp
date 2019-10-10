// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: MNIST loader.

#include "mnistloader.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ios;

using namespace ftensor;


void EndianConvert(int &data)
{
	unsigned char data_byte[4];
	data_byte[0] = data & 255;
	data_byte[1] = (data >> 8) & 255;
	data_byte[2] = (data >> 16) & 255;
	data_byte[3] = (data >> 24) & 255;
	data = ((int)data_byte[0] << 24) | ((int)data_byte[1] << 16) | ((int)data_byte[2] << 8) | data_byte[3];
}


void LoadMnistLabels(string label_file_name, vector<float>&labels)
{
	const char* label_file_name_c = label_file_name.data();
	ifstream label_file(label_file_name_c, ios::binary);
	if (label_file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		label_file.read((char*)&magic_number, sizeof(magic_number));
		label_file.read((char*)&number_of_images, sizeof(number_of_images));
		EndianConvert(magic_number);
		EndianConvert(number_of_images);

		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			label_file.read((char*)&label, sizeof(label));
			labels.push_back((float)label);
		}

	}
}


void LoadMnistImages(string image_file_name, vector<vector<float> >&images)
{
	const char* image_file_name_c = image_file_name.data();
	ifstream image_file(image_file_name_c, ios::binary);
	if (image_file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		image_file.read((char*)&magic_number, sizeof(magic_number));
		image_file.read((char*)&number_of_images, sizeof(number_of_images));
		image_file.read((char*)&n_rows, sizeof(n_rows));
		image_file.read((char*)&n_cols, sizeof(n_cols));
		EndianConvert(magic_number);
		EndianConvert(number_of_images);
		EndianConvert(n_rows);
		EndianConvert(n_cols);

		for (int i = 0; i < number_of_images; i++)
		{
			vector<float>tp;
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					image_file.read((char*)&image, sizeof(image));
					tp.push_back(image);
				}
			}
			images.push_back(tp);
		}
	}
}


void LoadDatabase(const std::string& path, Tensor& training_images, Tensor& training_labels, Tensor& testing_images, Tensor& testing_labels) {

	vector<vector<float> >* training_image_temp = new vector<vector<float> >;
	LoadMnistImages(path + "/train-images.idx3-ubyte", *training_image_temp);
	training_images = Reshape(Tensor(*training_image_temp), 28, 28, 60000);
	if (nullptr != training_image_temp) {
		delete training_image_temp;
		training_image_temp = nullptr;
	}

	vector<vector<float> >* testing_image_temp = new vector<vector<float> >;
	LoadMnistImages(path + "/t10k-images.idx3-ubyte", *testing_image_temp);
	testing_images = Reshape(Tensor(*testing_image_temp), 28, 28, 10000);
	if (nullptr != testing_image_temp) {
		delete testing_image_temp;
		testing_image_temp = nullptr;
	}

	vector<float>* training_label_temp = new vector<float>;
	LoadMnistLabels(path + "/train-labels.idx1-ubyte", *training_label_temp);
	training_labels = Transpose(Tensor(*training_label_temp)) + 1;
	if (nullptr != training_label_temp) {
		delete training_label_temp;
		training_label_temp = nullptr;
	}

	vector<float>* testing_label_temp = new vector<float>;
	LoadMnistLabels(path + "/t10k-labels.idx1-ubyte", *testing_label_temp);
	testing_labels = Transpose(Tensor(*testing_label_temp)) + 1;
	if (nullptr != testing_label_temp) {
		delete testing_label_temp;
		testing_label_temp = nullptr;
	}

	training_images.Info();
	training_labels.Info();
	testing_images.Info();
	testing_labels.Info();
}
