// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: GOKU chip.

#ifndef __GOKU_H__
#define __GOKU_H__


#include <iostream>

#include "tensorlib.h"
#include "device.h"
#include "drivers/jyn_spi.h"
#include "drivers/jyn_gpio.h"
#include "drivers/jyn_flash.h"


class Goku :public Device {
public:
	Goku();
	virtual ~Goku();

	void UseSimulator(bool option = true);
	void InitSimulator();
	void ShutDownSimulator();

	void AllOnes();
	void AllOnesSim();
	void AllZeros();
	void AllZerosSim();
	void BlockErase(int wl_id);
	void BlockEraseSim(int wl_id);
	bool BlockProgramAndVerify(int wl_id, itensor32::Tensor* pointer_to_a_tensor_3240);
	bool BlockProgramAndVerifySim(int wl_id, itensor32::Tensor* pointer_to_a_tensor_3240);
	void ExecuteAndRead(itensor32::Tensor* pointer_to_a_tensor_324);
	void ExecuteAndReadSim(itensor32::Tensor* pointer_to_a_tensor_324);
	void LoadData(itensor32::Tensor* pointer_to_a_tensor_400);
	void LoadDataSim(itensor32::Tensor* pointer_to_a_tensor_400);

private:
	bool spi_working_;
	itensor32::Tensor goku_array_;
	itensor32::Tensor input_buffer_;
	itensor32::Tensor output_buffer_;
	itensor32::Tensor weight_buffer_;
};


#endif  // __GOKU_H__
