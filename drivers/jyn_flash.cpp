#include "jyn_flash.h"


static int32_t input_buffer[TOTAL_WL_NUM] = {0,};			// Input buffer for the 400 columns
static int32_t output_buffer[TOTAL_OUTPUT_NUM] = {0,};		// Output buffer for the 324 rows
static int32_t expected_output[TOTAL_OUTPUT_NUM] = {0,};		// Expected outputs for a given block

// Address lists for WLs and BLs
static uint32_t bl_group_names[TOTAL_BL_GROUP_NUM] = {0x00,};
static uint8_t wl_group_names[TOTAL_WL_GROUP_NUM] = {0x00,};
static uint8_t wl_names[8] = {0x00,};


uint8_t FlashReadDeviceId() {
	// Initialize name lists
	for (uint32_t i = 0; i < TOTAL_WL_GROUP_NUM; ++i) {
		wl_group_names[i] = i;
	}
	for (uint32_t i = 0; i < 8; ++i) {
		wl_names[i] = 0x01 << i;
	}
	uint32_t Address_Code_List[9] = {0x00, 0x02, 0x04, 0x08, 0x0A, 0x0C, 0x10, 0x12, 0x14};
	for (uint32_t j = 0; j < 9; ++j) {
		for (uint32_t i = 0; i < 9; ++i) {
			bl_group_names[j * 9 + i] = (Address_Code_List[j] << 4) + Address_Code_List[i];
		}
	}


	uint8_t temp = 0;
	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_DEVICE_ID);
	SpiSendByte(DUMMY_BYTE);
	temp = SpiSendByte(DUMMY_BYTE);
	SPI_FLASH_CS_HIGH();
	return temp;
}


uint8_t FlashReadByte() {
	return (SpiSendByte(DUMMY_BYTE));
}


void FlashDelay(uint32_t usec) {
	usleep(usec);
}


void FlashExecute() {
	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_EXECUTE);
	SpiSendByte((FLASH_EXECUTE_ADDRESS & 0xFF0000) >> 16);
	SpiSendByte((FLASH_EXECUTE_ADDRESS & 0x00FF00) >> 8);
	SpiSendByte(FLASH_EXECUTE_ADDRESS & 0x0000FF);
	SpiSendByte(DUMMY_BYTE);
	SPI_FLASH_CS_HIGH();
	FlashDelay(DELAY_EXECUTE);
}


void FlashReadData(uint32_t read_address) {
	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_READ_DATA);
	SpiSendByte((read_address & 0xFF0000) >> 16);
	SpiSendByte((read_address & 0x00FF00) >> 8);
	SpiSendByte(read_address & 0x0000FF);
	SpiSendByte(DUMMY_BYTE);
}


void FlashBlockErase(int wl_id) {
	int WL_Group_Index = wl_id / 8;
	int WL_Index = wl_id % 8;
	uint8_t wl_group_name = wl_group_names[WL_Group_Index];
	uint8_t wl_name = wl_names[WL_Index];

	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_LOAD_DATA);
	SpiSendByte(0x00);
	SpiSendByte(0x00);
	SpiSendByte(wl_group_name);
	SpiSendByte(wl_name);
	SPI_FLASH_CS_HIGH();

	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_BLOCK_ERASE);
	SpiSendByte(0x00);
	SpiSendByte(0x00);
	SpiSendByte(0x00);
	SPI_FLASH_CS_HIGH();
	FlashDelay(DELAY_BLOCK_ERASE);
}


void FlashBlockProgram(int wl_id, int32_t* pointer_to_an_int32_t_array_3240) {
	int32_t* ptr = pointer_to_an_int32_t_array_3240;
	for (int bl_group_index = 0; bl_group_index < TOTAL_BL_GROUP_NUM; ++bl_group_index) {
		uint8_t bytes[5] = {0x00,};
		for (int iter_byte = 0; iter_byte < 5; ++iter_byte) {
			for (int iter_bit = 0; iter_bit < 8; ++iter_bit) {
				bytes[iter_byte] |= ((*ptr)? (0x01 << iter_bit): 0x00);
				ptr++;
			}
		}
		FlashPageProgram(wl_id, bl_group_index, bytes[4], bytes[3], bytes[2], bytes[1], bytes[0]);
	}
}


void FlashPageProgram(int wl_id, int bl_group_index, uint8_t byte4, uint8_t byte3, uint8_t byte2, uint8_t byte1, uint8_t byte0) {
	int WL_Group_Index = wl_id / 8;
	int WL_Index = wl_id % 8;
	uint8_t wl_group_name = wl_group_names[WL_Group_Index];
	uint8_t wl_name = wl_names[WL_Index];
	uint32_t bl_group_name = bl_group_names[bl_group_index];

#ifdef FIVE_FOLD_MODE
	// 5-iteration programming method (much higher success rate)
	FlashOneProgramOperation(wl_group_name, wl_name, bl_group_name, byte4 & 0x03, byte3 & 0x0C, byte2 & 0x30, byte1 & 0xC0, byte0 & 0x00);
	FlashOneProgramOperation(wl_group_name, wl_name, bl_group_name, byte4 & 0x0C, byte3 & 0x30, byte2 & 0xC0, byte1 & 0x00, byte0 & 0x03);
	FlashOneProgramOperation(wl_group_name, wl_name, bl_group_name, byte4 & 0x30, byte3 & 0xC0, byte2 & 0x00, byte1 & 0x03, byte0 & 0x0C);
	FlashOneProgramOperation(wl_group_name, wl_name, bl_group_name, byte4 & 0xC0, byte3 & 0x00, byte2 & 0x03, byte1 & 0x0C, byte0 & 0x30);
	FlashOneProgramOperation(wl_group_name, wl_name, bl_group_name, byte4 & 0x00, byte3 & 0x03, byte2 & 0x0C, byte1 & 0x30, byte0 & 0xC0);
#else
	// 2-iteration programming method
	FlashOneProgramOperation(wl_group_name, wl_name, bl_group_name, byte4 & 0xF0, byte3 & 0xF0, byte2 & 0xF0, byte1 & 0xF0, byte0 & 0xF0);
	FlashOneProgramOperation(wl_group_name, wl_name, bl_group_name, byte4 & 0x0F, byte3 & 0x0F, byte2 & 0x0F, byte1 & 0x0F, byte0 & 0x0F);
#endif
}


// Please do NOT call this function outside FlashPageProgram
void FlashOneProgramOperation(uint8_t wl_group_name, uint8_t wl_name, uint32_t bl_group_name, \
	uint8_t byte4, uint8_t byte3, uint8_t byte2, uint8_t byte1, uint8_t byte0) {
	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_LOAD_DATA);

	SpiSendByte(0x00);
	SpiSendByte(0x00);
	SpiSendByte(wl_group_name);

	SpiSendByte(wl_name);
	SPI_FLASH_CS_HIGH();

	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_PAGE_PROGRAM);

	SpiSendByte((bl_group_name & 0xFF0000) >> 16);
	SpiSendByte((bl_group_name & 0x00FF00) >> 8);
	SpiSendByte(bl_group_name & 0x0000FF);

	SpiSendByte(byte4);
	SpiSendByte(byte3);
	SpiSendByte(byte2);
	SpiSendByte(byte1);
	SpiSendByte(byte0);

	SPI_FLASH_CS_HIGH();
	FlashDelay(DELAY_PAGE_PROGRAM);
}


void FlashExecuteAndRead(int32_t* pointer_to_an_int32_t_array_324) {
	int32_t* ptr = pointer_to_an_int32_t_array_324;
	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_EXECUTE);
	SpiSendByte((FLASH_EXECUTE_ADDRESS & 0xFF0000) >> 16);
	SpiSendByte((FLASH_EXECUTE_ADDRESS & 0x00FF00) >> 8);
	SpiSendByte(FLASH_EXECUTE_ADDRESS & 0x0000FF);
	SpiSendByte(DUMMY_BYTE);
	SPI_FLASH_CS_HIGH();
	FlashDelay(DELAY_EXECUTE);

	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_READ_DATA);
	SpiSendByte((FLASH_READ_ADDRESS & 0xFF0000) >> 16);
	SpiSendByte((FLASH_READ_ADDRESS & 0x00FF00) >> 8);
	SpiSendByte(FLASH_READ_ADDRESS & 0x0000FF);
	SpiSendByte(DUMMY_BYTE);

	uint8_t temp = 0;
	if (nullptr == ptr) {
		for (int j = 0; j < 18; ++j) {
			for (int i = 0; i < 9; ++i) {
				temp = SpiSendByte(DUMMY_BYTE);
				temp = ((temp & 0xF0) >> 4) | ((temp & 0x0F) << 4);
				printf("%02x ", temp);
			}
			printf("\n");
		}
		printf("\n");
	}
	else {
		for (int j = 0; j < 18; ++j) {
			for (int i = 0; i < 9; ++i) {
				temp = SpiSendByte(DUMMY_BYTE);
				*ptr = int32_t(temp & 0x0F) - 5;
				ptr++;
				*ptr = int32_t((temp & 0xF0) >> 4) - 5;
				ptr++;
			}
		}
	}
	SPI_FLASH_CS_HIGH();
}


void FlashAllZeros() {
	for (int wl_id = 0; wl_id < TOTAL_WL_NUM; ++wl_id) {
		for (int bl_group_index = 0; bl_group_index < TOTAL_BL_GROUP_NUM; ++bl_group_index) {
			FlashPageProgram(wl_id, bl_group_index, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
		}
	}
}


void FlashAllOnes() {
	for (int wl_id = 0; wl_id < TOTAL_WL_NUM; ++wl_id) {
		FlashBlockErase(wl_id);
	}
}


void FlashLoadData(int32_t* pointer_to_an_int32_t_array_400) {
	int32_t* ptr = pointer_to_an_int32_t_array_400;
	SPI_FLASH_CS_LOW();
	SpiSendByte(GOKU_LOAD_DATA);
	SpiSendByte((FLASH_READ_ADDRESS & 0xFF0000) >> 16);
	SpiSendByte((FLASH_READ_ADDRESS & 0x00FF00) >> 8);
	SpiSendByte(FLASH_READ_ADDRESS & 0x0000FF);
	for (int j = 0; j < TOTAL_WL_GROUP_NUM; ++j) {
		uint8_t byte0 = 0x00;
		for (int i = 0; i < 8; ++i) {
			byte0 |= ((*ptr)? (0x01 << i): 0x00);
			ptr++;
		}
		SpiSendByte(byte0);
	}
	SPI_FLASH_CS_HIGH();
	FlashDelay(DELAY_LOAD_DATA);
}


bool FlashBlockProgramAndVerify(int wl_id, int32_t* pointer_to_an_int32_t_array_3240) {
	bool all_correct;
	int32_t* ptr = pointer_to_an_int32_t_array_3240;
	memset(input_buffer, 0, sizeof(input_buffer));  // All set to 0
	input_buffer[wl_id] = 1;  // Only select the target WL

	// Work out the expected outputs
	memset(expected_output, 0, sizeof(expected_output));  // All set to 0
	for (int output_iter = 0; output_iter < TOTAL_OUTPUT_NUM; ++output_iter) {
		for (int line_iter = 0; line_iter < 5; ++line_iter, ++ptr) {
			expected_output[output_iter] += (*ptr)? 1: 0;
		}
		for (int line_iter = 5; line_iter < 10; ++line_iter, ++ptr) {
			expected_output[output_iter] -= (*ptr)? 1: 0;
		}
	}

	all_correct = true;

	// Verify
	FlashLoadData(input_buffer);
	FlashExecuteAndRead(output_buffer);
	for (int output_iter = 0; output_iter < TOTAL_OUTPUT_NUM; ++output_iter) {
		if (output_buffer[output_iter] != expected_output[output_iter]) {
			all_correct = false;
			break;
		}
	}
	if (all_correct) {
		return true;
	}
	else {
		all_correct = true;

		FlashBlockErase(wl_id);

		FlashBlockErase(wl_id);
		
		ptr = pointer_to_an_int32_t_array_3240;
		for (int page_id = 0; page_id < TOTAL_BL_GROUP_NUM; ++page_id) {
			uint8_t bytes[5] = {0x00,};
			for (int iter_byte = 0; iter_byte < 5; ++iter_byte) {
				for (int iter_bit = 0; iter_bit < 8; ++iter_bit) {
					bytes[iter_byte] |= ((*ptr)? (0x01 << iter_bit): 0x00);
					ptr++;
				}
			}
			bool page_all_correct;
			for (int retry_time = 0; retry_time < MAXIMUM_PROGRAM_RETRY; ++retry_time) {
				page_all_correct = true;
				FlashPageProgram(wl_id, page_id, bytes[4], bytes[3], bytes[2], bytes[1], bytes[0]);

				// Verify
				FlashLoadData(input_buffer);
				FlashExecuteAndRead(output_buffer);
				for (int xxx = 0; xxx < 4; ++xxx) {
					int pos = xxx + page_id * 4;
					if (output_buffer[pos] != expected_output[pos]) {
						page_all_correct = false;
						break;
					}
				}
				if (page_all_correct) {
					break;
				}
			}
			if (!page_all_correct) {
				all_correct = false;
			}
		}
	}
	return all_correct;
}
