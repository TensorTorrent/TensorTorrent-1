// Author: Yuning Jiang
// Date: Oct. 2 nd, 2019
// Description: Includes and definitions for tensorlib.h

#ifndef __TENSORLIB_DEFS_H__
#define __TENSORLIB_DEFS_H__


#include <iostream>
#include <cstring>
#include <cstdlib>
#include <random>
#include <fstream>
#include <vector>
#include <typeinfo>
#include <iomanip>
#include <chrono>
#include <assert.h>
//#include <immintrin.h>

#define TENSOR_DEBUG

// Macro functions
#define T_CHECK_TYPE(T) assert(typeid(float).name() == typeid(T).name() || typeid(int32_t).name() == typeid(T).name())

#define T_WARNING(MESSAGE) std::cout << "Warning: " << MESSAGE << std::endl

#define T_ERROR(MESSAGE) std::cerr << "Error: " << MESSAGE << std::endl; exit(1)

#define UPDATE_DIMS_AND_NUMEL() dims_ = (rows_ > 1) + (cols_ > 1) + (slis_ > 1) + (gros_ > 1); numel_ = rows_ * cols_ * slis_ * gros_

#define INT32_T_MOD(A, B) (A % B)

#define INT32_T_SIGN(A) ((A > 0)? 1: ((A < 0)? -1: 0))

#define FLOAT_SIGN(A) ((A > 0.0)? 1.0: ((A < 0.0)? -1.0: 0.0))

#define INCREMENT(A) (A++)

#define DECREMENT(A) (A--)

#define ADD_OPERATION(A, B) (A + B)

#define MINUS_OPERATION(A, B) (A - B)

#define INT32_T_SHIFT_LEFT(A, B) (A << B)

#define INT32_T_SHIFT_RIGHT(A, B) (A >> B)

#define BITWISE_COMPLEMENT(A) (~A)

#define BITWISE_AND(A, B) (A & B)

#define BITWISE_OR(A, B) (A | B)

#define BITWISE_NAND(A, B) (~(A & B))

#define BITWISE_NOR(A, B) (~(A | B))

#define BITWISE_EXCLUSIVE_OR(A, B) (A ^ B)

#define BITWISE_EXCLUSIVE_NOR(A, B) (~(A ^ B))

#define GREATER_THAN(A, B) (A > B)

#define LESS_THAN(A, B) (A < B)

#define EQUAL_TO(A, B) (A == B)

#define NOT_LESS_THAN(A, B) (A >= B)

#define NOT_GREATER_THAN(A, B) (A <= B)

#define NOT_EQUAL_TO(A, B) (A != B)

#define NOT_EQUAL_TO_ZERO(A) (0 != A)

#define DECLARE_FUNC_T(MyFunc) \
Tensor MyFunc(const Tensor& ts)

#define DEFINE_FUNC_T(MyFunc, Func) \
Tensor MyFunc(const Tensor& ts) { \
	int numel_ts = ts.numel(); \
	if (0 == numel_ts) { \
		return ts; \
	} \
	else { \
		Tensor a; \
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros()); \
		auto data_a = a.data(); \
		auto data_ts = ts.data(); \
		for (int n = 0; n < numel_ts; ++n) { \
			data_a[n] = Func(data_ts[n]); \
		} \
		return a; \
	} \
}

#define DECLARE_FUNC_T_S(MyFunc, TDAT) \
TensorTemplate<TDAT> MyFunc(const TensorTemplate<TDAT>& ts, const TDAT& num)

#define DEFINE_FUNC_T_S(MyFunc, Func, TDAT) \
TensorTemplate<TDAT> MyFunc(const TensorTemplate<TDAT>& ts, const TDAT& num) { \
	int numel_ts = ts.numel(); \
	if (0 == numel_ts) { \
		return ts; \
	} \
	else { \
		TensorTemplate<TDAT> a; \
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros()); \
		auto data_a = a.data(); \
		auto data_ts = ts.data(); \
		for (int n = 0; n < numel_ts; ++n) { \
			data_a[n] = Func(data_ts[n], num); \
		} \
		return a; \
	} \
}

#define DECLARE_FUNC_S_T(MyFunc, TDAT) \
TensorTemplate<TDAT> MyFunc(const TDAT& num, const TensorTemplate<TDAT>& ts)

#define DEFINE_FUNC_S_T(MyFunc, Func, TDAT) \
TensorTemplate<TDAT> MyFunc(const TDAT& num, const TensorTemplate<TDAT>& ts) { \
	int numel_ts = ts.numel(); \
	if (0 == numel_ts) { \
		return ts; \
	} \
	else { \
		TensorTemplate<TDAT> a; \
		a.Resize(ts.rows(), ts.cols(), ts.slis(), ts.gros()); \
		auto data_a = a.data(); \
		auto data_ts = ts.data(); \
		for (int n = 0; n < numel_ts; ++n) { \
			data_a[n] = Func(num, data_ts[n]); \
		} \
		return a; \
	} \
}

#define DECLARE_FUNC_T_T(MyFunc, TDAT) \
TensorTemplate<TDAT> MyFunc(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& b)

#define DEFINE_FUNC_T_T(MyFunc, Func, TDAT) \
TensorTemplate<TDAT> MyFunc(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& b) { \
	if (MatchTemplate<TDAT>(a, b)) { \
		int numel_a = a.numel(); \
		if (0 == numel_a) { \
			return a; \
		} \
		else { \
			TensorTemplate<TDAT> c; \
			c.Resize(a.rows(), a.cols(), a.slis(), a.gros()); \
			auto data_a = a.data(); \
			auto data_b = b.data(); \
			auto data_c = c.data(); \
			for (int n = 0; n < numel_a; ++n) { \
				data_c[n] = Func(data_a[n], data_b[n]); \
			} \
			return c; \
		} \
	} \
	else { \
		T_ERROR("Dimension mismatched.\n"); \
		return a; \
	} \
}

#define DECLARE_FUNC_O_I(MyFunc, OutType, InType) \
TensorTemplate<OutType> MyFunc(const TensorTemplate<InType>& a)

#define DEFINE_FUNC_O_I(MyFunc, Func, OutType, InType) \
TensorTemplate<OutType> MyFunc(const TensorTemplate<InType>& a) { \
	int numel_a = a.numel(); \
	TensorTemplate<OutType> c; \
	c.Resize(a.rows(), a.cols(), a.slis(), a.gros()); \
	if (0 == numel_a) { \
		return c; \
	} \
	else { \
		auto data_a = a.data(); \
		auto data_c = c.data(); \
		for (int n = 0; n < numel_a; ++n) { \
			data_c[n] = Func(data_a[n]); \
		} \
		return c; \
	} \
}

#define DECLARE_FUNC_O_I_I(MyFunc, OutType, InTypeA, InTypeB) \
TensorTemplate<OutType> MyFunc(const TensorTemplate<InTypeA>& a, const TensorTemplate<InTypeB>& b)

#define DEFINE_FUNC_O_I_I(MyFunc, Func, OutType, InTypeA, InTypeB) \
TensorTemplate<OutType> MyFunc(const TensorTemplate<InTypeA>& a, const TensorTemplate<InTypeB>& b) { \
	TensorTemplate<OutType> c; \
	if (a.rows() == b.rows() && a.cols() == b.cols() && a.slis() == b.slis() && a.gros() == b.gros()) { \
		int numel_a = a.numel(); \
		c.Resize(a.rows(), a.cols(), a.slis(), a.gros()); \
		if (0 == numel_a) { \
			return c; \
		} \
		else { \
			auto data_a = a.data(); \
			auto data_b = b.data(); \
			auto data_c = c.data(); \
			for (int n = 0; n < numel_a; ++n) { \
				data_c[n] = Func(data_a[n], data_b[n]); \
			} \
			return c; \
		} \
	} \
	else { \
		T_ERROR("Dimension mismatched.\n"); \
		return c; \
	} \
}


struct BITMAPFILEHEADER {
	uint32_t bfSize;
	uint16_t bfReserved1;
	uint16_t bfReserved2;
	uint32_t bfOffBits;
};


struct BITMAPINFOHEADER {
	uint32_t biSize;
	uint32_t biWidth;
	uint32_t biHeight;
	uint16_t biPlanes;
	uint16_t biBitCount;
	uint32_t biCompression;
	uint32_t biSizeImage;
	uint32_t biXPelsPerMeter;
	uint32_t biYPelsPerMeter;
	uint32_t biClrUsed; 
	uint32_t biClrImportant;
};


struct RGB32 {
	uint8_t rgbBlue;
	uint8_t rgbGreen;
	uint8_t rgbRed;
	uint8_t rgbReserved;
};


struct RGB24 {
	uint8_t rgbBlue;
	uint8_t rgbGreen;
	uint8_t rgbRed;
};


struct RGB16 {
	uint16_t rgbData;
};


#endif  // __TENSORLIB_DEFS_H__
