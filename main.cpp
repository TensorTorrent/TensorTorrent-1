#include <iostream>
#include <ctime>
#include "tensortorrent.h"

#include "drivers/jyn_spi.h"
#include "drivers/jyn_gpio.h"
#include "drivers/jyn_flash.h"
#include "goku.h"
#include "conv2dgoku.h"
#include "convtranspose2dgoku.h"


using namespace itensor32;
using std::string;
using std::vector;
using std::cout;
using std::endl;


int main() {
	const bool SIM = false;
	const int PADDING = 1;
	Goku goku_chip_1;

	clock_t start, end;
	start = clock();
	auto hkernel = Tensor("-1, 0, 1; -1, 0, 1; -1, 0, 1");
	// Load the MNIST database
	ftensor::Tensor training_images, training_labels, testing_images, testing_labels;
	
	LoadDatabase("./examples/data/mnist", training_images, training_labels, testing_images, testing_labels);
	auto image_1 = training_images.S(-1, -1, -1, -1, 0, 1);
	Tensor input_image = Tensor(ftensor::Round(ftensor::AvgPool2d(image_1.S(2, 26, 2, 26))));
	cout << "\nInput Image:\n" << endl;
	input_image.Show();
/*
	auto after_conv2d_1 = Conv2d(input_image, hkernel, 1, PADDING);
	cout << "\nAfter Conv2d (s = 1):\n" << endl;
	after_conv2d_1.Show();

	Conv2dGoku conv_1(1, 1, 3, 1, PADDING);
	conv_1.to(&goku_chip_1, SIM);
	auto after_goku_conv2d_1 = conv_1(input_image);
	cout << "\nAfter GokuConv2d (s = 1):\n" << endl;
	after_goku_conv2d_1.Show();

	auto after_conv2d_2 = Conv2d(input_image, hkernel, 2, PADDING);
	cout << "\nAfter Conv2d (s = 2):\n" << endl;
	after_conv2d_2.Show();

	Conv2dGoku conv_2(1, 1, 3, 2, PADDING);
	conv_2.to(&goku_chip_1, SIM);
	auto after_goku_conv2d_2 = conv_2(input_image);
	cout << "\nAfter GokuConv2d (s = 2):\n" << endl;
	after_goku_conv2d_2.Show();
*/
	auto after_convtranspose2d_1 = ConvTranspose2d(input_image, hkernel, 1, PADDING);
	cout << "\nAfter ConvTranspose2d (s = 1):\n" << endl;
	after_convtranspose2d_1.Show();
SaveTensor("conv.bin", after_convtranspose2d_1);

	ConvTranspose2dGoku convtranspose_1(1, 1, 3, 1, PADDING);
	convtranspose_1.to(&goku_chip_1, SIM);
	auto after_goku_convtranspose2d_1 = convtranspose_1(input_image);
	cout << "\nAfter GokuConvTranspose2d (s = 1):\n" << endl;
	after_goku_convtranspose2d_1.Show();
SaveTensor("convgoku.bin", after_goku_convtranspose2d_1);
/*
	auto after_convtranspose2d_2 = ConvTranspose2d(input_image, hkernel, 2, PADDING);
	cout << "\nAfter ConvTranspose2d (s = 2):\n" << endl;
	after_convtranspose2d_2.Show();

	ConvTranspose2dGoku convtranspose_2(1, 1, 3, 2, PADDING);
	convtranspose_2.to(&goku_chip_1, SIM);
	auto after_goku_convtranspose2d_2 = convtranspose_2(input_image);
	cout << "\nAfter GokuConvTranspose2d (s = 2):\n" << endl;
	after_goku_convtranspose2d_2.Show();
*/

	end = clock();
	printf("Use Time:%f s\n", ((double)(end - start) / CLOCKS_PER_SEC));
	return 0;
}
