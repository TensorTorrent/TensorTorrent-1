// Author: Yuning Jiang
// Date: Oct. 1 st, 2019
// Description: Tensor library.
// Supports 0-D (scalars), 1-D (vectors), 2-D (arrays), 3-D (cubes), and 4-D tensors.

#include "tensorlib.h"


using std::vector;
using std::string;


static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
static std::default_random_engine generator (seed);
static const int maxn = 256;
static int m[maxn][maxn];


static void OddMagic(int n) {
	memset(m, 0, sizeof(m));
	int x = 0, y = n / 2;
	for (int i = 1; i <= n * n; ++i) {
		m[x][y] = i;
		x--;
		y++;
		if (x < 0 && y > n - 1) {
			x = x + 2;
			y = y - 1;
		}
		else if (x < 0)
			x = x + n;
		else if (y > n - 1)
			y = y - n;
		else if (0 != m[x][y]) {
			x = x + 2;
			y = y - 1;
		}
	}
}


static void DoubleEvenMagic(int n) {
	memset(m, 0, sizeof(m));
	for (int i = 1, x = 0, y = 0; i <= n * n; ++i) {
		m[x][y] = i;
		y++;
		if (y > n - 1) {
			x++;
			y -= n;
		}
	}
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (0 == i % 4 && 0 == j % 4) {
				for (int k = 0; k < 4; ++k)
					m[i + k][j + k] = (n * n + 1) - m[i + k][j + k];
			}
			else if (3 == i % 4 && 0 == j % 4) {
				for (int k = 0; k < 4; ++k)
					m[i - k][j + k] = (n * n + 1) - m[i - k][j + k];
			}
		}
	}
}


static void SingleEvenMagic(int n) {
	memset(m, 0, sizeof(m));
	int n0 = n / 2;
	OddMagic(n0);
	for (int i = 0; i < n0; ++i) {
		for (int j = 0; j < n0; ++j) {
			m[i + n0][j + n0] = m[i][j] + n0 * n0;
			m[i][j + n0] = m[i + n0][j + n0] + n0 * n0;
			m[i + n0][j] = m[i][j + n0] + n0 * n0;
		}
	}
	int k = (n - 2) / 4;
	for (int i = 0; i < n0; ++i) {
		for (int j = 0; j < k; ++j) {
			if (i == n0 / 2) {
				std::swap(m[i][i + j], m[i + n0][i + j]);
			}
			else {
				std::swap(m[i][j], m[i + n0][j]);
			}
			for (int i=0; i < n0; i++) {
				for (int j = n0 + n0 / 2; j > n0 + n0 / 2 - (k - 1); --j) {
					std::swap(m[i][j], m[i + n0][j]);
				}
			}
		}
	}
}


static bool VerifyMagic(int n) {
	int cnt = n * (n * n + 1) / 2;
	for (int i = 0; i < n; ++i) {
		int sum_row = 0, sum_line = 0;
		for (int j = 0; j < n; ++j) {
			sum_row += m[i][j];
			sum_line += m[j][i];
		}
		if (sum_row != cnt || sum_line != cnt) {
			return false;
		}
	}
	int sum_left = 0, sum_right = 0;
	for (int i = 0; i < n; ++i) {
		sum_left += m[i][i];
		sum_right += m[n - i - 1][i];
	}
	if (sum_left != cnt || sum_right != cnt) {
		return false;
	}
	return true;
}


static void WrapMagic(int n, TensorTemplate<int32_t>& ts) {
	ts.Resize(n, n);
	int32_t* data_ts = ts.data();
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			*data_ts++ = m[i][j];
		}
	}
}


static void WrapMagic(int n, TensorTemplate<float>& ts) {
	ts.Resize(n, n);
	float* data_ts = ts.data();
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			*data_ts++ = m[i][j];
		}
	}
}


static int32_t IntPower(int32_t a, int32_t b) {
	int32_t result = 1;
	for (int32_t i = 0; i < b; ++i) {
		result *= a;
	}
	return result;
}


void SplitTensorString(const string& src, const string& separator, vector<string>& dest)
{
	string str = src;
	string substring;
	string::size_type start = 0, index;

	do {
		index = str.find_first_of(separator, start);
		if (index != string::npos) {
			substring = str.substr(start, index-start);
			dest.push_back(substring);
			start = str.find_first_not_of(separator, index);
			if (start == string::npos) return;
		}
	} while (index != string::npos);

	substring = str.substr(start);
	dest.push_back(substring);
}


vector<int> RandomIndex(int n) {
	vector<int> v(n);
	vector<int> res;
	for (int i = 0; i < n; ++i) {
		v[i] = i;
	}
	srand(unsigned(time(NULL)));
	for (int i = n; i>0; i--) {
		int index = rand() % i;
		res.push_back(v[index]);
		v.erase(v.begin() + index);
	}
	return res;
}


TensorTemplate<int32_t> LoadBmp(const string& file_name, const string& option) {
	TensorTemplate<int32_t> ts;
	BITMAPFILEHEADER bmp_header;
	BITMAPINFOHEADER bmp_info;
	int rows_ts = 0;
	int cols_ts = 0;
	int slis_ts = 3;
	int gros_ts = 1;
	int area_ts = 0;
	bool greyscale = false;
	if (option == "greyscale") {
		slis_ts = 1;
		greyscale = true;
	}
	std::ifstream input_file(file_name,std::ios::in | std::ios::binary);
	if(input_file) {
		uint16_t bfType;
		input_file.read((char *)&bfType, sizeof(uint16_t));
		if (0x4d42 != bfType) {
			#ifdef TENSOR_DEBUG
			T_WARNING("The file is not a bmp file.\n");
			#endif
			return ts;
		}
		input_file.read((char*)&bmp_header, sizeof(BITMAPFILEHEADER));
		input_file.read((char*)&bmp_info, sizeof(BITMAPINFOHEADER));
		if (0 != bmp_info.biClrUsed) {
			#ifdef TENSOR_DEBUG
			T_WARNING("Unsupported bmp format (with pallet).\n");
			#endif
			return ts;
		}
		rows_ts = bmp_info.biHeight;
		cols_ts = bmp_info.biWidth;
		area_ts = rows_ts * cols_ts;
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		int32_t* data_ts = ts.data();
		input_file.seekg(bmp_header.bfOffBits, std::ios::beg);
		if (bmp_info.biBitCount == 32) {
			uint32_t real_width = bmp_info.biWidth << 2;
			uint8_t* temp = new uint8_t[real_width]();
			for (uint32_t i = 0; i < bmp_info.biHeight; ++i) {
				input_file.read((char *)temp, real_width);
				RGB32* rgb_data = (RGB32*)temp;
				for(uint32_t j = 0; j < bmp_info.biWidth; ++ j) {
					if (greyscale) {
						data_ts[(rows_ts - i - 1) * cols_ts + j] = int32_t(rgb_data[j].rgbRed * 0.299 + \
						rgb_data[j].rgbGreen * 0.587 + rgb_data[j].rgbBlue * 0.114);
					}
					else {
						data_ts[(rows_ts - i - 1) * cols_ts + j] = int32_t(rgb_data[j].rgbRed);
						data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts +j] = int32_t(rgb_data[j].rgbGreen);
						data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts +j] = int32_t(rgb_data[j].rgbBlue);
					}
				}
				rgb_data = nullptr;
			}
			if (nullptr != temp) {
				delete [] temp;
				temp = nullptr;
			}
		}
		else if (bmp_info.biBitCount == 24) {
			uint32_t real_width = ((bmp_info.biWidth * 24 + 31) >> 5) << 2;
			uint8_t* temp = new uint8_t[real_width]();
			for (uint32_t i = 0; i < bmp_info.biHeight; ++i) {
				input_file.read((char *)temp, real_width);
				RGB24* rgb_data = (RGB24*)temp;
				for(uint32_t j = 0; j < bmp_info.biWidth; ++j) {
					if (greyscale) {
						data_ts[(rows_ts - i - 1) * cols_ts +j] = int32_t(rgb_data[j].rgbRed * 0.299 + \
						rgb_data[j].rgbGreen * 0.587 + rgb_data[j].rgbBlue * 0.114);
					}
					else {
						data_ts[(rows_ts - i - 1) * cols_ts +j] = int32_t(rgb_data[j].rgbRed);
						data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts +j] = int32_t(rgb_data[j].rgbGreen);
						data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts +j] = int32_t(rgb_data[j].rgbBlue);
					}
				}
				rgb_data = nullptr;
			}
			if (nullptr != temp) {
				delete [] temp;
				temp = nullptr;
			}
		}
		else if (bmp_info.biBitCount == 16) {
			uint32_t real_width = ((bmp_info.biWidth * 16 + 31) >> 5) << 2;
			uint8_t* temp = new uint8_t[real_width]();
			for (uint32_t i = 0; i < bmp_info.biHeight; ++i) {
				input_file.read((char *)temp, real_width);
				RGB16* rgb_data = (RGB16*)temp;
				for(uint32_t j = 0; j < bmp_info.biWidth; ++j) {
					int32_t red_data = rgb_data[j].rgbData & 0x1F;
					int32_t green_data = (rgb_data[j].rgbData >> 5) & 0x1F;
					int32_t blue_data = (rgb_data[j].rgbData >> 10) & 0x1F;
					if (greyscale) {
						data_ts[(rows_ts - i - 1) * cols_ts +j] = int32_t(red_data * 0.299 + green_data * 0.587 + blue_data * 0.114);
					}
					else {
						data_ts[(rows_ts - i - 1) * cols_ts +j] = blue_data;
						data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts +j] = green_data;
						data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts +j] = red_data;
					}
				}
				rgb_data = nullptr;
			}
			if (nullptr != temp) {
				delete [] temp;
				temp = nullptr;
			}
		}
		else {
			#ifdef TENSOR_DEBUG
			T_WARNING("Unsupported Colors.\n");
			#endif
		}
		input_file.close();
		return ts;
	}
	else{
		#ifdef TENSOR_DEBUG
		T_WARNING("Unable to open the file.\n");
		#endif
		return ts;
	}
}


bool SaveBmp(const string& file_name, const TensorTemplate<int32_t>& ts, const string& option, uint16_t color_bit) {
	int rows_ts = ts.rows();
	int cols_ts = ts.cols();
	int slis_ts = ts.slis();
	int gros_ts = ts.gros();
	int numel_ts = ts.numel();
	bool greyscale = false;
	if (option == "greyscale") {
		greyscale = true;
	}
	if (0 == numel_ts) {
		#ifdef TENSOR_DEBUG
		T_WARNING("The tensor is empty. Nothing to be saved.\n");
		#endif
		return false;
	}
	if (1 != gros_ts) {
		#ifdef TENSOR_DEBUG
		T_WARNING("The size of the fourth dimension must be 1.\n");
		#endif
		return false;
	}
	if (1 != slis_ts && 3 != slis_ts) {
		#ifdef TENSOR_DEBUG
		T_WARNING("The size of the third dimension must be either 1 or 3.\n");
		#endif
		return false;
	}
	std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
	if(!output_file) {
		#ifdef TENSOR_DEBUG
		T_WARNING("Unable to create the file.\n");
		#endif
		return false;
	}
	else {
		BITMAPFILEHEADER bmp_header;
		BITMAPINFOHEADER bmp_info;
		if (color_bit == 32 || color_bit == 24 || color_bit == 16) {
			int32_t* data_ts = ts.data();
			uint32_t real_width = (((uint32_t)cols_ts * color_bit + 31) >> 5) << 2;
			int32_t area_ts = rows_ts * cols_ts;
			bmp_header.bfReserved1 = 0;
			bmp_header.bfReserved2 = 0;
			bmp_header.bfOffBits = 54;
			bmp_info.biSize = 40;
			bmp_info.biWidth = cols_ts;
			bmp_info.biHeight = rows_ts;
			bmp_info.biPlanes = 1;
			bmp_info.biBitCount = color_bit;
			bmp_info.biCompression = 0;
			bmp_info.biSizeImage = rows_ts * real_width + 2;
			bmp_info.biXPelsPerMeter = 11811;
			bmp_info.biYPelsPerMeter = 11811;
			bmp_info.biClrUsed = 0;
			bmp_info.biClrImportant = 0;
			bmp_header.bfSize = bmp_info.biSizeImage + 54;
			uint16_t bfType = 0x4d42;
			output_file.write((char*)&bfType, sizeof(uint16_t));
			output_file.write((char*)&bmp_header, sizeof(BITMAPFILEHEADER));
			output_file.write((char*)&bmp_info, sizeof(BITMAPINFOHEADER));
			if (color_bit == 32) {
				uint8_t* temp = new uint8_t[real_width]();
				if (1 == slis_ts) {
					for(int i = 0; i < rows_ts; ++i) {
						RGB32* temp_ptr = (RGB32*)temp;
						for(int j = 0; j < cols_ts; ++j) {
							uint8_t grey_data = uint8_t(data_ts[(rows_ts - i - 1) * cols_ts + j]);
							RGB32 rgb_data;
							rgb_data.rgbBlue = grey_data;
							rgb_data.rgbGreen = grey_data;
							rgb_data.rgbRed = grey_data;
							rgb_data.rgbReserved = 0;
							*(temp_ptr++) = rgb_data;
						}
						output_file.write((char*)temp, real_width);
					}
				}
				else {
					if (greyscale) {
						for(int i = 0; i < rows_ts; ++i) {
							RGB32* temp_ptr = (RGB32*)temp;
							for(int j = 0; j < cols_ts; ++j) {
								int32_t grey_temp = (299 * data_ts[(rows_ts - i - 1) * cols_ts + j] + \
									587 * data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts + j] + \
									114 * data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts + j]) / 1000;
								uint8_t grey_data = uint8_t(grey_temp);
								RGB32 rgb_data;
								rgb_data.rgbBlue = grey_data;
								rgb_data.rgbGreen = grey_data;
								rgb_data.rgbRed = grey_data;
								rgb_data.rgbReserved = 0;
								*(temp_ptr++) = rgb_data;
							}
							output_file.write((char*)temp, real_width);
						}
					}
					else {
						for(int i = 0; i < rows_ts; ++i) {
							RGB32* temp_ptr = (RGB32*)temp;
							for(int j = 0; j < cols_ts; ++j) {
								RGB32 rgb_data;
								rgb_data.rgbBlue = uint8_t(data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts + j]);
								rgb_data.rgbGreen = uint8_t(data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts + j]);
								rgb_data.rgbRed = uint8_t(data_ts[(rows_ts - i - 1) * cols_ts + j]);
								rgb_data.rgbReserved = 0;
								*(temp_ptr++) = rgb_data;
							}
							output_file.write((char*)temp, real_width);
						}
					}
				}
				if (nullptr != temp) {
					delete [] temp;
					temp = nullptr;
				}
			}
			else if (color_bit == 24) {
				uint8_t* temp = new uint8_t[real_width]();
				if (1 == slis_ts) {
					for(int i = 0; i < rows_ts; ++i) {
						RGB24* temp_ptr = (RGB24*)temp;
						for(int j = 0; j < cols_ts; ++j) {
							uint8_t grey_data = uint8_t(data_ts[(rows_ts - i - 1) * cols_ts + j]);
							RGB24 rgb_data;
							rgb_data.rgbBlue = grey_data;
							rgb_data.rgbGreen = grey_data;
							rgb_data.rgbRed = grey_data;
							*(temp_ptr++) = rgb_data;
						}
						output_file.write((char*)temp, real_width);
					}
				}
				else {
					if (greyscale) {
						for(int i = 0; i < rows_ts; ++i) {
							RGB24* temp_ptr = (RGB24*)temp;
							for(int j = 0; j < cols_ts; ++j) {
								int32_t grey_temp = (299 * data_ts[(rows_ts - i - 1) * cols_ts + j] + \
									587 * data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts + j] + \
									114 * data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts + j]) / 1000;
								uint8_t grey_data = uint8_t(grey_temp);
								RGB24 rgb_data;
								rgb_data.rgbBlue = grey_data;
								rgb_data.rgbGreen = grey_data;
								rgb_data.rgbRed = grey_data;
								*(temp_ptr++) = rgb_data;
							}
							output_file.write((char*)temp, real_width);
						}
					}
					else {
						for(int i = 0; i < rows_ts; ++i) {
							RGB24* temp_ptr = (RGB24*)temp;
							for(int j = 0; j < cols_ts; ++j) {
								RGB24 rgb_data;
								rgb_data.rgbBlue = uint8_t(data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts + j]);
								rgb_data.rgbGreen = uint8_t(data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts + j]);
								rgb_data.rgbRed = uint8_t(data_ts[(rows_ts - i - 1) * cols_ts + j]);
								*(temp_ptr++) = rgb_data;
							}
							output_file.write((char*)temp, real_width);
						}
					}
				}
				if (nullptr != temp) {
					delete [] temp;
					temp = nullptr;
				}
			}
			else {
				uint8_t* temp = new uint8_t[real_width]();
				if (1 == slis_ts) {
					for(int i = 0; i < rows_ts; ++i) {
						RGB16* temp_ptr = (RGB16*)temp;
						for(int j = 0; j < cols_ts; ++j) {
							uint8_t grey_data = uint8_t(data_ts[(rows_ts - i - 1) * cols_ts + j]);
							RGB16 rgb_data;
							rgb_data.rgbData = ((grey_data & 0x1F) << 10) | ((grey_data & 0x1F) << 5) | ((grey_data & 0x1F));
							*(temp_ptr++) = rgb_data;
						}
						output_file.write((char*)temp, real_width);
					}
				}
				else {
					if (greyscale) {
						for(int i = 0; i < rows_ts; ++i) {
							RGB16* temp_ptr = (RGB16*)temp;
							for(int j = 0; j < cols_ts; ++j) {
								int32_t grey_temp = (299 * data_ts[(rows_ts - i - 1) * cols_ts + j] + \
									587 * data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts + j] + \
									114 * data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts + j]) / 1000;
								uint8_t grey_data = uint8_t(grey_temp);
								RGB16 rgb_data;
								rgb_data.rgbData = ((grey_data & 0x1F) << 10) | ((grey_data & 0x1F) << 5) | ((grey_data & 0x1F));
								*(temp_ptr++) = rgb_data;
							}
							output_file.write((char*)temp, real_width);
						}
					}
					else {
						for(int i = 0; i < rows_ts; ++i) {
							RGB16* temp_ptr = (RGB16*)temp;
							for(int j = 0; j < cols_ts; ++j) {
								RGB16 rgb_data;
								rgb_data.rgbData = ((uint8_t(data_ts[(rows_ts - i - 1) * cols_ts + j]) & 0x1F) << 10) | \
								((uint8_t(data_ts[area_ts * 1 + (rows_ts - i - 1) * cols_ts + j]) & 0x1F) << 5) | \
								((uint8_t(data_ts[area_ts * 2 + (rows_ts - i - 1) * cols_ts + j]) & 0x1F));
								*(temp_ptr++) = rgb_data;
							}
							output_file.write((char*)temp, real_width);
						}
					}
				}
				if (nullptr != temp) {
					delete [] temp;
					temp = nullptr;
				}
			}
		}
		else {
			#ifdef TENSOR_DEBUG
			T_WARNING("Unsupported Colors.\n");
			#endif
			return false;
		}
		output_file.close();
		return true;
	}
}


namespace itensor32 {


Tensor Raw(int rows, int cols, int slis, int gros) {
	return RawTemplate<int32_t>(rows, cols, slis, gros);
}


Tensor Raw(const Tensor& a) {
	return RawTemplate<int32_t>(a);
}


Tensor Zeros(int rows, int cols, int slis, int gros) {
	return ZerosTemplate<int32_t>(rows, cols, slis, gros);
}


Tensor Zeros(const Tensor& a) {
	return ZerosTemplate<int32_t>(a);
}


Tensor Ones(int rows, int cols, int slis, int gros) {
	return OnesTemplate<int32_t>(rows, cols, slis, gros);
}


Tensor Ones(const Tensor& a) {
	return OnesTemplate<int32_t>(a);
}


Tensor Arange(int num) {
	return ArangeTemplate<int32_t>(num);
}


bool Match(const Tensor& a, const Tensor& b) {
	return MatchTemplate<int32_t>(a, b);
}


int Numel(const Tensor& a) {
	return NumelTemplate<int32_t>(a);
}


TensorTemplate<int32_t> Size(const Tensor& a) {
	return SizeTemplate<int32_t>(a);
}


Tensor Reshape(const Tensor& a, int rows, int cols, int slis, int gros) {
	return ReshapeTemplate<int32_t>(a, rows, cols, slis, gros);
}


Tensor Transpose(const Tensor& a) {
	return TransposeTemplate<int32_t>(a);
}


Tensor Flip(const Tensor& a, int dim) {
	return FlipTemplate<int32_t>(a, dim);
}


Tensor Flip(const Tensor& a, const string& dim_string) {
	return FlipTemplate<int32_t>(a, dim_string);
}


Tensor Repmat(const Tensor& a, int rt, int ct, int st, int gt) {
	return RepmatTemplate<int32_t>(a, rt, ct, st, gt);
}


Tensor Kron(const Tensor& a, const Tensor& b) {
	return KronTemplate<int32_t>(a, b);
}


Tensor Permute(const Tensor& a, int dim0, int dim1, int dim2, int dim3) {
	return PermuteTemplate<int32_t>(a, dim0, dim1, dim2, dim3);
}


Tensor Rot90(const Tensor& a, int times, int axis) {
	return Rot90Template<int32_t>(a, times, axis);
}


Tensor Rearrange(const Tensor& a, const vector<int>& v, int dim) {
	return RearrangeTemplate<int32_t>(a, v, dim);
}


Tensor Sum(const Tensor& a) {
	return SumTemplate<int32_t>(a);
}


Tensor Sum(const Tensor& a, int dim) {
	return SumTemplate<int32_t>(a, dim);
}


Tensor Mean(const Tensor& a) {
	return MeanTemplate<int32_t>(a);
}


Tensor Mean(const Tensor& a, int dim) {
	return MeanTemplate<int32_t>(a, dim);
}


Tensor Stddev(const Tensor& a, const std::string& ddof) {
	return StddevTemplate<int32_t>(a, ddof);
}


Tensor Stddev(const Tensor& a, int dim, const std::string& ddof) {
	return StddevTemplate<int32_t>(a, dim, ddof);
}


Tensor Var(const Tensor& a, const std::string& ddof) {
	return VarTemplate<int32_t>(a, ddof);
}


Tensor Var(const Tensor& a, int dim, const std::string& ddof) {
	return VarTemplate<int32_t>(a, dim, ddof);
}


Tensor Max(const Tensor& a) {
	return MaxTemplate<int32_t>(a);
}


Tensor Max(const Tensor& a, int dim, Tensor* pos) {
	return MaxTemplate<int32_t>(a, dim, pos);
}


Tensor Min(const Tensor& a) {
	return MinTemplate<int32_t>(a);
}


Tensor Min(const Tensor& a, int dim, Tensor* pos) {
	return MinTemplate<int32_t>(a, dim, pos);
}


Tensor operator*(const Tensor& a, const Tensor& b) {
	return MMTemplate<int32_t>(a, b);
}


Tensor MM(const Tensor& a, const Tensor& b) {
	return MMTemplate<int32_t>(a, b);
}


Tensor Where(const Tensor& a, const Tensor& b, const Tensor& c) {
	return WhereTemplate<int32_t>(a, b, c);
}


Tensor Cat(const vector<Tensor>& v, int dim) {
	return CatTemplate<int32_t>(v, dim);
}


vector<Tensor> Split(const Tensor& a, int dim) {
	return SplitTemplate<int32_t>(a, dim);
}


Tensor PaddingAsym(const Tensor& a, int ra, int rb, int ca, int cb, int sa, int sb, int ga, int gb, int32_t padding_value) {
	return PaddingAsymTemplate<int32_t>(a, ra, rb, ca, cb, sa, sb, ga, gb, padding_value);
}


Tensor Padding(const Tensor& a, int ra, int ca, int sa, int ga, int32_t padding_value) {
	return PaddingAsymTemplate<int32_t>(a, ra, ra, ca, ca, sa, sa, ga, ga, padding_value);
}


Tensor AvgPool2d(const Tensor& a, int k, Tensor* mask) {
	return AvgPool2dTemplate<int32_t>(a, k, mask);
}


Tensor MaxPool2d(const Tensor& a, int k, Tensor* mask) {
	return MaxPool2dTemplate<int32_t>(a, k, mask);
}


Tensor Conv2dBase(const Tensor& a, const Tensor& k) {
	return Conv2dBaseTemplate<int32_t>(a, k);
}


Tensor Conv2d(const Tensor& a, const Tensor& k, int stride, int padding) {
	return Conv2dTemplate<int32_t>(a, k, stride, padding);
}


Tensor ConvTranspose2d(const Tensor& a, const Tensor& k, int stride, int padding) {
	return ConvTranspose2dTemplate<int32_t>(a, k, stride, padding);
}


void WriteTensor(std::ofstream& output_file, Tensor* ts) {
	WriteTensorTemplate<int32_t>(output_file, ts);
}


void ReadTensor(std::ifstream& input_file, Tensor* ts) {
	ReadTensorTemplate<int32_t>(input_file, ts);
}


void SaveTensor(string file_name, Tensor* ts) {
	SaveTensorTemplate<int32_t>(file_name, ts);
}


void SaveTensors(string file_name, vector<Tensor>* tensor_group) {
	SaveTensorsTemplate<int32_t>(file_name, tensor_group);
}


Tensor LoadTensor(string file_name) {
	return LoadTensorTemplate<int32_t>(file_name);
}


vector<Tensor> LoadTensors(string file_name) {
	return LoadTensorsTemplate<int32_t>(file_name);
}


Tensor Magic(int rows) {
	if (rows < 3) {
		switch (rows) {
			case 1:
			return Tensor(1);
			case 2:
			return Tensor("1, 3; 4, 2");
			default:
			return Tensor();
		}
	}
	else {
		Tensor ts;
		if (rows & 1) {
			OddMagic(rows);
			if (VerifyMagic(rows)) {
				WrapMagic(rows, ts);
			}
		}
		else if (!(rows % 4)) {
			DoubleEvenMagic(rows);
			if (VerifyMagic(rows)) {
				WrapMagic(rows, ts);
			}
		}
		else {
			SingleEvenMagic(rows);
			if (VerifyMagic(rows)) {
				WrapMagic(rows, ts);
			}
		}
		return ts;
	}
}


Tensor Rand(int rows, int cols, int slis, int gros, int32_t upper, int32_t lower) {
	double data_upper = upper;
	double data_lower = lower;
	if (upper < lower) {
		data_upper = lower;
		data_lower = upper;
	}
	Tensor ts = RawTemplate<int32_t>(rows, cols, slis, gros);
	std::uniform_int_distribution<int32_t> distribution(data_lower, data_upper);
	int32_t* data_ts = ts.data();
	int numel_ts = ts.numel();
	for (int i = 0; i < numel_ts; ++i) {
		data_ts[i] = distribution(generator);
	}
	return ts;
}


DEFINE_FUNC_T(Abs, abs)
DEFINE_FUNC_T(Sign, INT32_T_SIGN)
DEFINE_FUNC_T_S(operator%, INT32_T_MOD, int32_t);
DEFINE_FUNC_S_T(operator%, INT32_T_MOD, int32_t);
DEFINE_FUNC_T_T(operator%, INT32_T_MOD, int32_t);
DEFINE_FUNC_T_S(Mod, INT32_T_MOD, int32_t);
DEFINE_FUNC_S_T(Mod, INT32_T_MOD, int32_t);
DEFINE_FUNC_T_T(Mod, INT32_T_MOD, int32_t);
DEFINE_FUNC_T_S(Pow, IntPower, int32_t);
DEFINE_FUNC_S_T(Pow, IntPower, int32_t);
DEFINE_FUNC_T_T(Pow, IntPower, int32_t);
DEFINE_FUNC_T_S(operator+, ADD_OPERATION, int32_t);
DEFINE_FUNC_S_T(operator+, ADD_OPERATION, int32_t);
DEFINE_FUNC_T_T(operator+, ADD_OPERATION, int32_t);
DEFINE_FUNC_T_S(operator-, MINUS_OPERATION, int32_t);
DEFINE_FUNC_S_T(operator-, MINUS_OPERATION, int32_t);
DEFINE_FUNC_T_T(operator-, MINUS_OPERATION, int32_t);
DEFINE_FUNC_T_S(Plus, ADD_OPERATION, int32_t);
DEFINE_FUNC_S_T(Plus, ADD_OPERATION, int32_t);
DEFINE_FUNC_T_T(Plus, ADD_OPERATION, int32_t);
DEFINE_FUNC_T_S(Minus, MINUS_OPERATION, int32_t);
DEFINE_FUNC_S_T(Minus, MINUS_OPERATION, int32_t);
DEFINE_FUNC_T_T(Minus, MINUS_OPERATION, int32_t);


Tensor operator*(const Tensor& ts, const int32_t& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] * num;
			}
		}
		return a;
	}
}


Tensor operator*(const int32_t& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = num * data_ts[n];
			}
		}
		return a;
	}
}


Tensor Mul(const Tensor& ts, const int32_t& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] * num;
			}
		}
		return a;
	}
}


Tensor Mul(const int32_t& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = num * data_ts[n];
			}
		}
		return a;
	}
}


Tensor Mul(const Tensor& a, const Tensor& b) {
	if (Match(a, b)) {
		int numel_a = a.numel();
		if (0 == numel_a) {
			return a;
		}
		else {
			Tensor c;
			c.Resize(a.rows(), a.cols(), a.slis(), a.gros());
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			for (int n = 0; n < numel_a; ++n) {
				data_c[n] = data_a[n] * data_b[n];
			}
			return c;
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


Tensor operator/(const Tensor& ts, const int32_t& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		if (0 == num) {
			T_ERROR("Division by zero.\n");
		}
		else if (1 == num) {
			return ts;
		}
		else {
			Tensor a;
			a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] / num;
			}
			return a;
		}
	}
}


Tensor operator/(const int32_t& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				if (0 == data_ts[n]) {
					T_ERROR("Division by zero.\n");
				}
				else {
					data_a[n] = num / data_ts[n];
				}
			}
		}
		return a;
	}
}


Tensor Div(const Tensor& ts, const int32_t& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		if (0 == num) {
			T_ERROR("Division by zero.\n");
		}
		else if (1 == num) {
			return ts;
		}
		else {
			Tensor a;
			a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] / num;
			}
			return a;
		}
	}
}


Tensor Div(const int32_t& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				if (0 == data_ts[n]) {
					T_ERROR("Division by zero.\n");
				}
				else {
					data_a[n] = num / data_ts[n];
				}
			}
		}
		return a;
	}
}


Tensor Div(const Tensor& a, const Tensor& b) {
	if (Match(a, b)) {
		int numel_a = a.numel();
		if (0 == numel_a) {
			return a;
		}
		else {
			Tensor c;
			c.Resize(a.rows(), a.cols(), a.slis(), a.gros());
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			for (int n = 0; n < numel_a; ++n) {
				if (0 == data_b[n]) {
					T_ERROR("Division by zero.\n");
				}
				else {
					data_c[n] = data_a[n] / data_b[n];
				}
			}
			return c;
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


DEFINE_FUNC_T_S(operator>, GREATER_THAN, int32_t);
DEFINE_FUNC_S_T(operator>, GREATER_THAN, int32_t);
DEFINE_FUNC_T_T(operator>, GREATER_THAN, int32_t);
DEFINE_FUNC_T_S(operator<, LESS_THAN, int32_t);
DEFINE_FUNC_S_T(operator<, LESS_THAN, int32_t);
DEFINE_FUNC_T_T(operator<, LESS_THAN, int32_t);
DEFINE_FUNC_T_S(operator==, EQUAL_TO, int32_t);
DEFINE_FUNC_S_T(operator==, EQUAL_TO, int32_t);
DEFINE_FUNC_T_T(operator==, EQUAL_TO, int32_t);
DEFINE_FUNC_T_S(operator>=, NOT_LESS_THAN, int32_t);
DEFINE_FUNC_S_T(operator>=, NOT_LESS_THAN, int32_t);
DEFINE_FUNC_T_T(operator>=, NOT_LESS_THAN, int32_t);
DEFINE_FUNC_T_S(operator<=, NOT_GREATER_THAN, int32_t);
DEFINE_FUNC_S_T(operator<=, NOT_GREATER_THAN, int32_t);
DEFINE_FUNC_T_T(operator<=, NOT_GREATER_THAN, int32_t);
DEFINE_FUNC_T_S(operator!=, NOT_EQUAL_TO, int32_t);
DEFINE_FUNC_S_T(operator!=, NOT_EQUAL_TO, int32_t);
DEFINE_FUNC_T_T(operator!=, NOT_EQUAL_TO, int32_t);
DEFINE_FUNC_T(Logic, NOT_EQUAL_TO_ZERO);
DEFINE_FUNC_T(operator++, INCREMENT);
DEFINE_FUNC_T(operator--, DECREMENT);
DEFINE_FUNC_T(operator-, NEGATIVE);
DEFINE_FUNC_T(IsNaN, IS_NAN);
DEFINE_FUNC_T(IsInf, IS_INF);
DEFINE_FUNC_T(IsFinite, IS_FINITE);


// For itensor32 only
DEFINE_FUNC_T(operator~, BITWISE_COMPLEMENT);
DEFINE_FUNC_T_S(operator<<, INT32_T_SHIFT_LEFT, int32_t);
DEFINE_FUNC_S_T(operator<<, INT32_T_SHIFT_LEFT, int32_t);
DEFINE_FUNC_T_T(operator<<, INT32_T_SHIFT_LEFT, int32_t);
DEFINE_FUNC_T_S(operator>>, INT32_T_SHIFT_RIGHT, int32_t);
DEFINE_FUNC_S_T(operator>>, INT32_T_SHIFT_RIGHT, int32_t);
DEFINE_FUNC_T_T(operator>>, INT32_T_SHIFT_RIGHT, int32_t);
DEFINE_FUNC_T_S(operator&, BITWISE_AND, int32_t);
DEFINE_FUNC_S_T(operator&, BITWISE_AND, int32_t);
DEFINE_FUNC_T_T(operator&, BITWISE_AND, int32_t);
DEFINE_FUNC_T_S(operator|, BITWISE_OR, int32_t);
DEFINE_FUNC_S_T(operator|, BITWISE_OR, int32_t);
DEFINE_FUNC_T_T(operator|, BITWISE_OR, int32_t);
DEFINE_FUNC_T_S(And, BITWISE_AND, int32_t);
DEFINE_FUNC_S_T(And, BITWISE_AND, int32_t);
DEFINE_FUNC_T_T(And, BITWISE_AND, int32_t);
DEFINE_FUNC_T_S(Or, BITWISE_OR, int32_t);
DEFINE_FUNC_S_T(Or, BITWISE_OR, int32_t);
DEFINE_FUNC_T_T(Or, BITWISE_OR, int32_t);
DEFINE_FUNC_T_S(Nand, BITWISE_NAND, int32_t);
DEFINE_FUNC_S_T(Nand, BITWISE_NAND, int32_t);
DEFINE_FUNC_T_T(Nand, BITWISE_NAND, int32_t);
DEFINE_FUNC_T_S(Nor, BITWISE_NOR, int32_t);
DEFINE_FUNC_S_T(Nor, BITWISE_NOR, int32_t);
DEFINE_FUNC_T_T(Nor, BITWISE_NOR, int32_t);
DEFINE_FUNC_T_S(Xor, BITWISE_EXCLUSIVE_OR, int32_t);
DEFINE_FUNC_S_T(Xor, BITWISE_EXCLUSIVE_OR, int32_t);
DEFINE_FUNC_T_T(Xor, BITWISE_EXCLUSIVE_OR, int32_t);
DEFINE_FUNC_T_S(Xnor, BITWISE_EXCLUSIVE_NOR, int32_t);
DEFINE_FUNC_S_T(Xnor, BITWISE_EXCLUSIVE_NOR, int32_t);
DEFINE_FUNC_T_T(Xnor, BITWISE_EXCLUSIVE_NOR, int32_t);


}  // namespace itensor32


namespace ftensor {


Tensor Raw(int rows, int cols, int slis, int gros) {
	return RawTemplate<float>(rows, cols, slis, gros);
}


Tensor Raw(const Tensor& a) {
	return RawTemplate<float>(a);
}


Tensor Zeros(int rows, int cols, int slis, int gros) {
	return ZerosTemplate<float>(rows, cols, slis, gros);
}


Tensor Zeros(const Tensor& a) {
	return ZerosTemplate<float>(a);
}


Tensor Ones(int rows, int cols, int slis, int gros) {
	return OnesTemplate<float>(rows, cols, slis, gros);
}


Tensor Ones(const Tensor& a) {
	return OnesTemplate<float>(a);
}


Tensor Arange(int num) {
	return ArangeTemplate<float>(num);
}


bool Match(const Tensor& a, const Tensor& b) {
	return MatchTemplate<float>(a, b);
}


int Numel(const Tensor& a) {
	return NumelTemplate<float>(a);
}


TensorTemplate<int32_t> Size(const Tensor& a) {
	return SizeTemplate<float>(a);
}


Tensor Reshape(const Tensor& a, int rows, int cols, int slis, int gros) {
	return ReshapeTemplate<float>(a, rows, cols, slis, gros);
}


Tensor Transpose(const Tensor& a) {
	return TransposeTemplate<float>(a);
}


Tensor Flip(const Tensor& a, int dim) {
	return FlipTemplate<float>(a, dim);
}


Tensor Flip(const Tensor& a, const string& dim_string) {
	return FlipTemplate<float>(a, dim_string);
}


Tensor Repmat(const Tensor& a, int rt, int ct, int st, int gt) {
	return RepmatTemplate<float>(a, rt, ct, st, gt);
}


Tensor Kron(const Tensor& a, const Tensor& b) {
	return KronTemplate<float>(a, b);
}


Tensor Permute(const Tensor& a, int dim0, int dim1, int dim2, int dim3) {
	return PermuteTemplate<float>(a, dim0, dim1, dim2, dim3);
}


Tensor Rot90(const Tensor& a, int times, int axis) {
	return Rot90Template<float>(a, times, axis);
}


Tensor Rearrange(const Tensor& a, const vector<int>& v, int dim) {
	return RearrangeTemplate<float>(a, v, dim);
}


Tensor Sum(const Tensor& a) {
	return SumTemplate<float>(a);
}


Tensor Sum(const Tensor& a, int dim) {
	return SumTemplate<float>(a, dim);
}


Tensor Mean(const Tensor& a) {
	return MeanTemplate<float>(a);
}


Tensor Mean(const Tensor& a, int dim) {
	return MeanTemplate<float>(a, dim);
}


Tensor Stddev(const Tensor& a, const std::string& ddof) {
	return StddevTemplate<float>(a, ddof);
}


Tensor Stddev(const Tensor& a, int dim, const std::string& ddof) {
	return StddevTemplate<float>(a, dim, ddof);
}


Tensor Var(const Tensor& a, const std::string& ddof) {
	return VarTemplate<float>(a, ddof);
}


Tensor Var(const Tensor& a, int dim, const std::string& ddof) {
	return VarTemplate<float>(a, dim, ddof);
}


Tensor Max(const Tensor& a) {
	return MaxTemplate<float>(a);
}


Tensor Max(const Tensor& a, int dim, Tensor* pos) {
	return MaxTemplate<float>(a, dim, pos);
}


Tensor Min(const Tensor& a) {
	return MinTemplate<float>(a);
}


Tensor Min(const Tensor& a, int dim, Tensor* pos) {
	return MinTemplate<float>(a, dim, pos);
}


Tensor operator*(const Tensor& a, const Tensor& b) {
	return MMTemplate<float>(a, b);
}


Tensor MM(const Tensor& a, const Tensor& b) {
	return MMTemplate<float>(a, b);
}


Tensor Where(const Tensor& a, const Tensor& b, const Tensor& c) {
	return WhereTemplate<float>(a, b, c);
}


Tensor Cat(const vector<Tensor>& v, int dim) {
	return CatTemplate<float>(v, dim);
}


vector<Tensor> Split(const Tensor& a, int dim) {
	return SplitTemplate<float>(a, dim);
}


Tensor PaddingAsym(const Tensor& a, int ra, int rb, int ca, int cb, int sa, int sb, int ga, int gb, float padding_value) {
	return PaddingAsymTemplate<float>(a, ra, rb, ca, cb, sa, sb, ga, gb, padding_value);
}


Tensor Padding(const Tensor& a, int ra, int ca, int sa, int ga, float padding_value) {
	return PaddingAsymTemplate<float>(a, ra, ra, ca, ca, sa, sa, ga, ga, padding_value);
}


Tensor AvgPool2d(const Tensor& a, int k, Tensor* mask) {
	return AvgPool2dTemplate<float>(a, k, mask);
}


Tensor MaxPool2d(const Tensor& a, int k, Tensor* mask) {
	return MaxPool2dTemplate<float>(a, k, mask);
}


Tensor Conv2dBase(const Tensor& a, const Tensor& k) {
	return Conv2dBaseTemplate<float>(a, k);
}


Tensor Conv2d(const Tensor& a, const Tensor& k, int stride, int padding) {
	return Conv2dTemplate<float>(a, k, stride, padding);
}


Tensor ConvTranspose2d(const Tensor& a, const Tensor& k, int stride, int padding) {
	return ConvTranspose2dTemplate<float>(a, k, stride, padding);
}


void WriteTensor(std::ofstream& output_file, Tensor* ts) {
	WriteTensorTemplate<float>(output_file, ts);
}


void ReadTensor(std::ifstream& input_file, Tensor* ts) {
	ReadTensorTemplate<float>(input_file, ts);
}


void SaveTensor(string file_name, Tensor* ts) {
	SaveTensorTemplate<float>(file_name, ts);
}


void SaveTensors(string file_name, vector<Tensor>* tensor_group) {
	SaveTensorsTemplate<float>(file_name, tensor_group);
}


Tensor LoadTensor(string file_name) {
	return LoadTensorTemplate<float>(file_name);
}


vector<Tensor> LoadTensors(string file_name) {
	return LoadTensorsTemplate<float>(file_name);
}


Tensor Magic(int rows) {
	if (rows < 3) {
		switch (rows) {
			case 1:
			return Tensor(1);
			case 2:
			return Tensor("1, 3; 4, 2");
			default:
			return Tensor();
		}
	}
	else {
		Tensor ts;
		if (rows & 1) {
			OddMagic(rows);
			if (VerifyMagic(rows)) {
				WrapMagic(rows, ts);
			}
		}
		else if (!(rows % 4)) {
			DoubleEvenMagic(rows);
			if (VerifyMagic(rows)) {
				WrapMagic(rows, ts);
			}
		}
		else {
			SingleEvenMagic(rows);
			if (VerifyMagic(rows)) {
				WrapMagic(rows, ts);
			}
		}
		return ts;
	}
}


Tensor Rand(int rows, int cols, int slis, int gros, float upper, float lower) {
	float data_upper = upper;
	float data_lower = lower;
	if (upper < lower) {
		data_upper = lower;
		data_lower = upper;
	}
	Tensor ts = RawTemplate<float>(rows, cols, slis, gros);
	std::uniform_real_distribution<float> distribution(data_lower, data_upper);
	float* data_ts = ts.data();
	int numel_ts = ts.numel();
	for (int n = 0; n < numel_ts; ++n) {
		data_ts[n] = distribution(generator);
	}
	return ts;
}


DEFINE_FUNC_T(Abs, abs)
DEFINE_FUNC_T(Sign, FLOAT_SIGN)
DEFINE_FUNC_T_S(operator%, fmod, float);
DEFINE_FUNC_S_T(operator%, fmod, float);
DEFINE_FUNC_T_T(operator%, fmod, float);
DEFINE_FUNC_T_S(Mod, fmod, float);
DEFINE_FUNC_S_T(Mod, fmod, float);
DEFINE_FUNC_T_T(Mod, fmod, float);
DEFINE_FUNC_T_S(Pow, pow, float);
DEFINE_FUNC_S_T(Pow, pow, float);
DEFINE_FUNC_T_T(Pow, pow, float);
DEFINE_FUNC_T_S(operator+, ADD_OPERATION, float);
DEFINE_FUNC_S_T(operator+, ADD_OPERATION, float);
DEFINE_FUNC_T_T(operator+, ADD_OPERATION, float);
DEFINE_FUNC_T_S(operator-, MINUS_OPERATION, float);
DEFINE_FUNC_S_T(operator-, MINUS_OPERATION, float);
DEFINE_FUNC_T_T(operator-, MINUS_OPERATION, float);
DEFINE_FUNC_T_S(Plus, ADD_OPERATION, float);
DEFINE_FUNC_S_T(Plus, ADD_OPERATION, float);
DEFINE_FUNC_T_T(Plus, ADD_OPERATION, float);
DEFINE_FUNC_T_S(Minus, MINUS_OPERATION, float);
DEFINE_FUNC_S_T(Minus, MINUS_OPERATION, float);
DEFINE_FUNC_T_T(Minus, MINUS_OPERATION, float);


Tensor operator*(const Tensor& ts, const float& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] * num;
			}
		}
		return a;
	}
}


Tensor operator*(const float& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = num * data_ts[n];
			}
		}
		return a;
	}
}


Tensor Mul(const Tensor& ts, const float& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] * num;
			}
		}
		return a;
	}
}


Tensor Mul(const float& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = num * data_ts[n];
			}
		}
		return a;
	}
}


Tensor Mul(const Tensor& a, const Tensor& b) {
	if (Match(a, b)) {
		int numel_a = a.numel();
		if (0 == numel_a) {
			return a;
		}
		else {
			Tensor c;
			c.Resize(a.rows(), a.cols(), a.slis(), a.gros());
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			for (int n = 0; n < numel_a; ++n) {
				data_c[n] = data_a[n] * data_b[n];
			}
			return c;
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


Tensor operator/(const Tensor& ts, const float& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		if (0 == num) {
			T_ERROR("Division by zero.\n");
		}
		else if (1 == num) {
			return ts;
		}
		else {
			Tensor a;
			a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] / num;
			}
			return a;
		}
	}
}


Tensor operator/(const float& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				if (0 == data_ts[n]) {
					T_ERROR("Division by zero.\n");
				}
				else {
					data_a[n] = num / data_ts[n];
				}
			}
		}
		return a;
	}
}


Tensor Div(const Tensor& ts, const float& num) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		if (0 == num) {
			T_ERROR("Division by zero.\n");
		}
		else if (1 == num) {
			return ts;
		}
		else {
			Tensor a;
			a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				data_a[n] = data_ts[n] / num;
			}
			return a;
		}
	}
}


Tensor Div(const float& num, const Tensor& ts) {
	int numel_ts = ts.numel();
	if (0 == numel_ts) {
		return ts;
	}
	else {
		Tensor a;
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros());
		if (0 == num) {
			a.Zeros();
		}
		else {
			auto data_a = a.data();
			auto data_ts = ts.data();
			for (int n = 0; n < numel_ts; ++n) {
				if (0 == data_ts[n]) {
					T_ERROR("Division by zero.\n");
				}
				else {
					data_a[n] = num / data_ts[n];
				}
			}
		}
		return a;
	}
}


Tensor Div(const Tensor& a, const Tensor& b) {
	if (Match(a, b)) {
		int numel_a = a.numel();
		if (0 == numel_a) {
			return a;
		}
		else {
			Tensor c;
			c.Resize(a.rows(), a.cols(), a.slis(), a.gros());
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			for (int n = 0; n < numel_a; ++n) {
				if (0 == data_b[n]) {
					T_ERROR("Division by zero.\n");
				}
				else {
					data_c[n] = data_a[n] / data_b[n];
				}
			}
			return c;
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


DEFINE_FUNC_T_S(operator>, GREATER_THAN, float);
DEFINE_FUNC_S_T(operator>, GREATER_THAN, float);
DEFINE_FUNC_T_T(operator>, GREATER_THAN, float);
DEFINE_FUNC_T_S(operator<, LESS_THAN, float);
DEFINE_FUNC_S_T(operator<, LESS_THAN, float);
DEFINE_FUNC_T_T(operator<, LESS_THAN, float);
DEFINE_FUNC_T_S(operator==, EQUAL_TO, float);
DEFINE_FUNC_S_T(operator==, EQUAL_TO, float);
DEFINE_FUNC_T_T(operator==, EQUAL_TO, float);
DEFINE_FUNC_T_S(operator>=, NOT_LESS_THAN, float);
DEFINE_FUNC_S_T(operator>=, NOT_LESS_THAN, float);
DEFINE_FUNC_T_T(operator>=, NOT_LESS_THAN, float);
DEFINE_FUNC_T_S(operator<=, NOT_GREATER_THAN, float);
DEFINE_FUNC_S_T(operator<=, NOT_GREATER_THAN, float);
DEFINE_FUNC_T_T(operator<=, NOT_GREATER_THAN, float);
DEFINE_FUNC_T_S(operator!=, NOT_EQUAL_TO, float);
DEFINE_FUNC_S_T(operator!=, NOT_EQUAL_TO, float);
DEFINE_FUNC_T_T(operator!=, NOT_EQUAL_TO, float);
DEFINE_FUNC_T(Logic, NOT_EQUAL_TO_ZERO);
DEFINE_FUNC_T(operator++, INCREMENT);
DEFINE_FUNC_T(operator--, DECREMENT);
DEFINE_FUNC_T(operator-, NEGATIVE);
DEFINE_FUNC_T(IsNaN, IS_NAN);
DEFINE_FUNC_T(IsInf, IS_INF);
DEFINE_FUNC_T(IsFinite, IS_FINITE);


// For ftensor only
Tensor Randn(int rows, int cols, int slis, int gros, float mean, float stddev) {
	Tensor ts = RawTemplate<float>(rows, cols, slis, gros);
	std::normal_distribution<float> distribution(mean, stddev);
	float* data_ts = ts.data();
	int numel_ts = ts.numel();
	for (int i = 0; i < numel_ts; ++i) {
		data_ts[i] = distribution(generator);
	}
	return ts;
}


DEFINE_FUNC_T(Sin, sin)
DEFINE_FUNC_T(Cos, cos)
DEFINE_FUNC_T(Tan, tan)
DEFINE_FUNC_T(Asin, asin)
DEFINE_FUNC_T(Acos, acos)
DEFINE_FUNC_T(Atan, atan)
DEFINE_FUNC_T(Sinh, sinh)
DEFINE_FUNC_T(Cosh, cosh)
DEFINE_FUNC_T(Tanh, tanh)
DEFINE_FUNC_T(Sqrt, sqrt)
DEFINE_FUNC_T(Ceil, ceil)
DEFINE_FUNC_T(Floor, floor)
DEFINE_FUNC_T(Round, round)
DEFINE_FUNC_T(Trunc, trunc)
DEFINE_FUNC_T(Log, log)
DEFINE_FUNC_T(Log10, log10)
DEFINE_FUNC_T(Exp, exp)

}  // namespace ftensor
