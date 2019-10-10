#include "jyn_spi.h"


// SPI settings
static uint8_t mode = DEFAULT_SPI_MODE;		// SPI mode. Use mode 0 as default
static uint8_t bits = DEFAULT_SPI_BITS;		// Bits per word. Use 8 as default
static uint32_t speed = DEFAULT_SPI_SPEED;	// Unit: Hz. NOT greater than 50000000
static uint16_t delay = DEFAULT_SPI_DELAY;	// Delay. It must be 0 or the SPI process goes wrong
static int fd;								// SPI file ID


static void SpiAbort(const char *s) {
	perror(s);
	abort();
}


uint8_t SpiSendByte(uint8_t data) {
	int ret;
	uint8_t result;
	uint8_t tx[] = {0x00,};
	uint8_t rx[ARRAY_SIZE(tx)] = {0x00,};
	tx[0] = data;
	struct spi_ioc_transfer tr = {
		.tx_buf = (unsigned long)tx,
		.rx_buf = (unsigned long)rx,
		.len = ARRAY_SIZE(tx),
		.delay_usecs = delay,
		.speed_hz = speed,
		.bits_per_word = bits,
	};
	ret = ioctl(fd, SPI_IOC_MESSAGE(1), &tr);
	if (ret < 1)
		SpiAbort("can't send spi message");
	
	result = rx[0];
	return result;
}


void SpiInit() {
	int ret = 0;
	fd = -1;

	fd = open(DEFAULT_DEVICE, O_RDWR);

	if (fd < 0)
		SpiAbort("can't open device");

	// SPI mode
	ret = ioctl(fd, SPI_IOC_WR_MODE, &mode);
	if (ret == -1)
		SpiAbort("can't set spi mode");
	ret = ioctl(fd, SPI_IOC_RD_MODE, &mode);
	if (ret == -1)
		SpiAbort("can't get spi mode");
	
	// Maximum speed (Hz)
	ret = ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);
	if (ret == -1)
		SpiAbort("can't set max speed hz");
	ret = ioctl(fd, SPI_IOC_RD_MAX_SPEED_HZ, &speed);
	if (ret == -1)
		SpiAbort("can't get max speed hz");

	// Bits per word
	ret = ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
	if (ret == -1)
		SpiAbort("can't set bits per word");
	ret = ioctl(fd, SPI_IOC_RD_BITS_PER_WORD, &bits);
	if (ret == -1)
		SpiAbort("can't get bits per word");

	printf("spi mode: %d\n", mode);
	printf("max speed: %d Hz (%.2f KHz)\n", speed, speed / 1000.0);
	printf("bits per word: %d\n", bits);
}


void SpiClose() {
	close(fd);
	fd = -1;
}


void SpiTest() {
	SpiInit();

	uint8_t ret = SpiSendByte(0xFD);
	printf("%.2X \n", ret);

	SpiClose();
}

