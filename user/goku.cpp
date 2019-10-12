// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: GOKU chip.

#include "goku.h"


const int WL_ADDRESS_BIAS = 1;


using namespace itensor32;


Goku::Goku()
: Device() {
	spi_working_ = false;
}


Goku::~Goku() {
	if (spi_working_) {
		SpiClose();
		spi_working_ = false;
	}
}


void Goku::UseSimulator(bool option) {
	if (option) {
		if (!use_simulator_) {
			InitSimulator();
			use_simulator_ = true;
		}
		if (spi_working_) {
			SpiClose();
			spi_working_ = false;
		}
	}
	else {
		if (use_simulator_) {
			ShutDownSimulator();
			use_simulator_ = false;
		}
		if (!spi_working_) {
			SpiInit();
			spi_working_ = true;
			auto device_id = FlashReadDeviceId();
			printf("Device ID: %02x\n\n", device_id);
//AllZeros();
			device_id = FlashReadDeviceId();
			printf("Device ID: %02x\n\n", device_id);
//AllZeros();
			device_id = FlashReadDeviceId();
			printf("Device ID: %02x\n\n", device_id);
		}
	}
}


void Goku::InitSimulator() {
	goku_array_ = Zeros(TOTAL_BL_NUM, TOTAL_WL_NUM);
	input_buffer_ = Zeros(1, TOTAL_WL_NUM);
	output_buffer_ = Zeros(1, TOTAL_OUTPUT_NUM);
	weight_buffer_ = Zeros(1, TOTAL_BL_NUM);
}


void Goku::ShutDownSimulator() {
	goku_array_.Clear();
	input_buffer_.Clear();
	output_buffer_.Clear();
	weight_buffer_.Clear();
}


void Goku::AllOnes() {
	if (use_simulator_) {
		return AllOnesSim();
	}
	else {
		return FlashAllOnes();
	}
}


void Goku::AllOnesSim() {
	goku_array_.Ones();
}


void Goku::AllZeros() {
	if (use_simulator_) {
		return AllZerosSim();
	}
	else {
		return FlashAllZeros();
	}
}


void Goku::AllZerosSim() {
	goku_array_.Zeros();
}


void Goku::BlockErase(int wl_id) {
	if (use_simulator_) {
		return BlockEraseSim(wl_id);
	}
	else {
		return FlashBlockErase(wl_id);
	}
}


void Goku::BlockEraseSim(int wl_id) {
	auto temp = Ones(3240, 1);
	goku_array_.S(temp, -1, -1, wl_id, wl_id + 1);
}


bool Goku::BlockProgramAndVerify(int wl_id, itensor32::Tensor* pointer_to_a_tensor_3240) {
	if (use_simulator_) {
		return BlockProgramAndVerifySim(wl_id, pointer_to_a_tensor_3240);
	}
	else {
		bool res = FlashBlockProgramAndVerify(wl_id, pointer_to_a_tensor_3240->data());
		if (res) std::cout << "WL #" << wl_id << ":  \tSuccess!" << std::endl;
		else std::cout << "WL #" << wl_id << ":  \tFailed!" << std::endl;
		return res;
	}
}


bool Goku::BlockProgramAndVerifySim(int wl_id, itensor32::Tensor* pointer_to_a_tensor_3240) {
BlockEraseSim(wl_id);
	auto temp = 1 - Reshape(*pointer_to_a_tensor_3240, TOTAL_BL_NUM, 1);
auto temp2 = goku_array_.S(-1, -1, wl_id, wl_id + 1);
	goku_array_.S((temp & temp2), -1, -1, wl_id, wl_id + 1);
	return true;
}


void Goku::ExecuteAndRead(itensor32::Tensor* pointer_to_a_tensor_324) {
	if (use_simulator_) {
		return ExecuteAndReadSim(pointer_to_a_tensor_324);
	}
	else {
		return FlashExecuteAndRead(pointer_to_a_tensor_324->data());
	}
}


void Goku::ExecuteAndReadSim(itensor32::Tensor* pointer_to_a_tensor_324) {
	pointer_to_a_tensor_324->Zeros();
	auto a = Logic(input_buffer_ * Transpose(goku_array_));
	int numel_a = a.numel();
	int32_t* data_a = a.data();
	int32_t* data_ts = pointer_to_a_tensor_324->data();
	for (int n = 0; n < numel_a; ++n) {
		int i_ts = n / 10;
		if (n % 10 < 5) {
			data_ts[i_ts] -= data_a[n];
		}
		else {
			data_ts[i_ts] += data_a[n];
		}
	}
}


void Goku::LoadData(itensor32::Tensor* pointer_to_a_tensor_400) {
	if (use_simulator_) {
		return LoadDataSim(pointer_to_a_tensor_400);
	}
	else {
		return FlashLoadData(pointer_to_a_tensor_400->data());
	}
}


void Goku::LoadDataSim(itensor32::Tensor* pointer_to_a_tensor_400) {
	input_buffer_ = *pointer_to_a_tensor_400;
}
