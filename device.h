// Author: Yuning Jiang
// Date: Oct. 6 th, 2019
// Description: Device.

#ifndef __DEVICE_H__
#define __DEVICE_H__


#include <iostream>

#include "tensorlib.h"


class Device {
public:
	Device();
	virtual ~Device();

	void UseSimulator(bool option = true) {use_simulator_ = option;}
	bool IsUsingSimulator() {return use_simulator_;}

protected:
	bool use_simulator_;
};


#endif  // __DEVICE_H__
