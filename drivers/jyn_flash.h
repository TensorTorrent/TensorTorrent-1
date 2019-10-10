// Copyright 2019 PKU
// Author: Yuning Jiang
// Date: Sep. 24th, 2019
// Description: SPI control using Linux drivers

#ifndef JYN_FLASH_H_
#define JYN_FLASH_H_

#include "jyn_spi.h"
#include "jyn_gpio.h"

#ifdef TPU
#define ADDR0 6
#define ADDR1 73
#else
#define ADDR0 27
#define ADDR1 23
#endif

// Use five-fold programming mode
#define FIVE_FOLD_MODE

// Basic operations
#define SPI_FLASH_CS_LOW() gpio_write(ADDR0, 0)
#define SPI_FLASH_CS_HIGH() gpio_write(ADDR0, 1)

// Commands
#define GOKU_RELEASE_POWER_DOWN	0xAB
#define GOKU_DEVICE_ID			0x9F
#define GOKU_EXECUTE			0x10
#define GOKU_GET_FEATURE		0x0F
#define GOKU_READ_DATA			0x0B
#define GOKU_LOAD_DATA			0x02
#define GOKU_BLOCK_ERASE		0xD8
#define GOKU_PAGE_PROGRAM		0x06
#define GOKU_MANUAL_PROGRAM		0x07
#define GOKU_WIP_FLAG			0x01
#define DUMMY_BYTE				0xFF

// Address
#define  FLASH_WRITE_ADDRESS	0x00000
#define  FLASH_EXECUTE_ADDRESS	0x00000
#define  FLASH_READ_ADDRESS		0x00000
#define  FLASH_LOAD_ADDRESS		0x00000

// Parameters
#define MAXIMUM_PROGRAM_RETRY	10
#define DEFAULT_DEVICE_ID		0xC8
#define TOTAL_BL_NUM			3240
#define TOTAL_WL_NUM			400
#define TOTAL_WL_GROUP_NUM		50
#define TOTAL_BL_GROUP_NUM		81
#define TOTAL_OUTPUT_NUM		324

// Delay
#define DELAY_EXECUTE			50
#define DELAY_BLOCK_ERASE		3000
#define DELAY_PAGE_PROGRAM		400
#define DELAY_LOAD_DATA			50
#define DELAY_PROGRAM_RETRY		10000


extern "C" {
uint8_t SpiSendByte(uint8_t data);
void SpiInit();
void SpiClose();
void SpiTest();
}


uint8_t FlashReadDeviceId();
uint8_t FlashReadByte();
void FlashDelay(uint32_t usec);
void FlashExecute();
void FlashReadData(uint32_t read_address);
void FlashBlockErase(int wl_id);
void FlashBlockProgram(int wl_id, int32_t* pointer_to_an_int32_t_array_3240);
void FlashPageProgram(int wl_id, int bl_group_index, uint8_t byte4, uint8_t byte3, uint8_t byte2, uint8_t byte1, uint8_t byte0);

// Please do NOT call this function outside FlashPageProgram
inline void FlashOneProgramOperation(uint8_t wl_group_name, uint8_t wl_name, uint32_t bl_group_name, \
	uint8_t byte4, uint8_t byte3, uint8_t byte2, uint8_t byte1, uint8_t byte0);

void FlashExecuteAndRead(int32_t* pointer_to_an_int32_t_array_324 = nullptr);
void FlashAllZeros();
void FlashAllOnes();
void FlashLoadData(int32_t* pointer_to_an_int32_t_array_400);
bool FlashBlockProgramAndVerify(int wl_id, int32_t* pointer_to_an_int32_t_array_3240);


#endif  // JYN_FLASH_H_
