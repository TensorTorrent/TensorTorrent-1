// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Convolution layer implemented by GOKU chips.

#include "conv2dgoku.h"


const int WL_ADDRESS_BIAS = 1;


using namespace itensor32;


Conv2dGoku::Conv2dGoku(int in_channels, int out_channels, int kernel_size, int stride, int padding)
: Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding) {
	kernel_value_map_ = Zeros(1, 9) - 1;
	kernel_ids_ = Zeros(1, 9) - 1;
}


Conv2dGoku::~Conv2dGoku() {
}


Tensor Conv2dGoku::WorkOutKernelMap(const Tensor& kernel) {
	Tensor k = kernel;
	kernel_value_map_ = Zeros(1, 9) - 1;
	int pp = 0;
	int nn = 0;
	for (int rk = 0; rk < 3; ++rk) {
		for (int ck = 0; ck < 3; ++ck) {
			if (1 == k(rk, ck)) {
				kernel_value_map_(3 * rk + ck) = pp + 5;
				pp++;
			}
			else if (-1 == k(rk, ck)) {
				kernel_value_map_(3 * rk + ck) = nn;
				nn++;
			}
			else {
				kernel_value_map_(3 * rk + ck) = -1;
			}
			if (pp > 4 || nn > 4) {
				break;
			}
		}
	}
	return kernel_value_map_;
}


Tensor Conv2dGoku::WorkOutKernelIds(int rpixel, int cpixel, int orows, int ocols, int stride, int padding) {
	kernel_ids_ = Zeros(1, 9) - 1;
	for (int i = 0; i < 3; ++i) {
		int ni = rpixel + padding - i;
		if (ni < 0) {
			continue;
		}
		else if (0 != ni % stride) {
			continue;
		}
		else {
			int ci = ni / stride + 1;
			if (ci > orows || ci < 1) {
				continue;
			}
			else {
				for (int j = 0; j < 3; ++j) {
					int nj = cpixel + padding - j;
					if (nj < 0) {
						continue;
					}
					else if (0 != nj % stride) {
						continue;
					}
					else {
						int cj = nj / stride + 1;
						if (cj > ocols || cj < 1) {
							continue;
						}
						else {
							kernel_ids_(3 * i + j) = ci * ocols + cj - ocols - 1;
						}
					}
				}
			}
		}
	}
	return kernel_ids_;
}


void Conv2dGoku::to(Goku* goku_chip, bool sim) {
	if (nullptr != goku_chip) {
		goku_chip_ = goku_chip;

		goku_chip_->UseSimulator(sim);

		//goku_chip_->AllOnes();

		auto hkernel = Tensor("-1, 0, 1; -1, 0, 1; -1, 0, 1");
		WorkOutKernelMap(hkernel);

		// Write the weights
		int ocols = ((12-3+padding_*2)/stride_)+1;
		int orows = ((12-3+padding_*2)/stride_)+1;

		auto weight_double_buffer = Zeros(1, TOTAL_BL_NUM * 2);
		for (int rpixel_iter = 0; rpixel_iter < 12; ++rpixel_iter) {
			for (int cpixel_iter = 0; cpixel_iter < 12; ++cpixel_iter) {
				weight_double_buffer.Zeros();
				WorkOutKernelIds(rpixel_iter, cpixel_iter, orows, ocols, stride_, padding_);
				for (int k = 0; k < 9; ++k) {
					if (kernel_value_map_(k) >= 0 && kernel_ids_(k) >= 0) {
						weight_double_buffer(kernel_ids_(k) * 10 + kernel_value_map_(k)) = 1;
					}
				}
				weight_double_buffer = 1 - weight_double_buffer;
				int wl_address_0 = WL_ADDRESS_BIAS + rpixel_iter * 12 + cpixel_iter;
				int wl_address_1 = wl_address_0 + 200;
				auto buf1 = weight_double_buffer.S(-1, -1, 0, 3240);
				auto buf2 = weight_double_buffer.S(-1, -1, 3240, 6480);
				goku_chip_->BlockProgramAndVerify(wl_address_0, &buf1);
				goku_chip_->BlockProgramAndVerify(wl_address_1, &buf2);
			}
		}
	}
	else {
		std::cerr << "Error: Empty goku pointer." << std::endl;
		exit(1);
	}
}


Tensor Conv2dGoku::operator()(const itensor32::Tensor& input_image) {
	int ocols = ((12-3+padding_*2)/stride_)+1;
	int orows = ((12-3+padding_*2)/stride_)+1;

	int32_t* data_in = input_image.data();
	// Reshape the input
	auto reshaped_input = Zeros(1, input_image.numel());
	for (int nrow = 0; nrow < 12; ++nrow) {
		for (int ncol = 0; ncol < 12; ++ncol) {
			reshaped_input(nrow * 12 + ncol) = data_in[nrow * 12 + ncol];
		}
	}
	auto distributed_input = Zeros(2, 400);
	distributed_input.S(reshaped_input, 0, 1, WL_ADDRESS_BIAS, 144 + WL_ADDRESS_BIAS);
	distributed_input.S(reshaped_input, 1, 2, 200 + WL_ADDRESS_BIAS, 344 + WL_ADDRESS_BIAS);

	// Calcultate
	auto final_result_buffer = Zeros(1, TOTAL_OUTPUT_NUM * 2);
	auto output_double_buffer = Zeros(1, TOTAL_OUTPUT_NUM * 2);
	for (int bit_iter = 0; bit_iter < 8; ++bit_iter) {
		for (int chunk_iter = 0; chunk_iter < 2; ++chunk_iter) {
			auto input_buffer = distributed_input.S(chunk_iter, chunk_iter + 1, -1, -1) % 2;
			auto temp = distributed_input.S(chunk_iter, chunk_iter + 1, -1, -1) / 2;
			distributed_input.S(temp, chunk_iter, chunk_iter + 1, -1, -1);
			goku_chip_->LoadData(&input_buffer);
			auto output_buffer = Zeros(1, TOTAL_OUTPUT_NUM);
			goku_chip_->ExecuteAndRead(&output_buffer);
			output_double_buffer.S(output_buffer, -1, -1, chunk_iter * 324, 324 + chunk_iter * 324);
		}
		final_result_buffer = final_result_buffer + output_double_buffer * int32_t(pow(2, bit_iter));
	}

	return Reshape(final_result_buffer.S(-1, -1, 0, (orows * ocols)), orows, ocols);
}
