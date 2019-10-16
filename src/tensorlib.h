// Author: Yuning Jiang
// Date: Oct. 1 st, 2019
// Description: Tensor library.
// Supports 0-D (scalars), 1-D (std::vectors), 2-D (arrays), 3-D (cubes), and 4-D tensors.

#ifndef __TENSORLIB_H__
#define __TENSORLIB_H__


#include "tensorlibdefs.h"


void SplitTensorString(const std::string& src, const std::string& separator, std::vector<std::string>& dest);
std::vector<int> RandomIndex(int n);


// ****************************************
// Tensor Template
// ****************************************
template <typename TDAT>
class TensorTemplate {
public:
	TensorTemplate();
	TensorTemplate(const TDAT& num);
	TensorTemplate(const std::vector<TDAT>& list);
	TensorTemplate(const std::vector<std::vector<TDAT> >& array);
	TensorTemplate(const std::vector<std::vector<std::vector<TDAT> > >& cube);
	TensorTemplate(const TensorTemplate<TDAT>& ts);
	TensorTemplate(const std::string& str);
	virtual ~TensorTemplate();

	int rows() const {return rows_;}
	int cols() const {return cols_;}
	int slis() const {return slis_;}
	int gros() const {return gros_;}
	int numel() const {return numel_;}
	int dims() const {return dims_;}
	TDAT* data() const {return data_;}

	TDAT Item();
	void Clear();
	int Size(int dim);
	void Resize(int rows = 1, int cols = 1, int slis = 1, int gros = 1);		// Will NOT keep the data
	TensorTemplate<TDAT> Reshape(int rows = 1, int cols = 1, int slis = 1, int gros = 1);		// Keep the data
	void Transpose();
	bool IsEmpty();
	void Zeros();
	void Ones();
	TensorTemplate<TDAT> Sum();
	void Info(std::ostream& out = std::cout);
	void Show(std::ostream& out = std::cout);
	TensorTemplate<TDAT> S(int ra = -1, int rb = -1, int ca = -1, int cb = -1, int sa = -1, int sb = -1, int ga = -1, int gb = -1);
	void S(const TensorTemplate<TDAT>& a, int ra = -1, int rb = -1, int ca = -1, int cb = -1, int sa = -1, int sb = -1, int ga = -1, int gb = -1);

	TDAT& operator()(int r = 0, int c = 0, int s = 0, int g = 0);

	void operator=(const TensorTemplate<TDAT>& ts);
	void operator+=(const TensorTemplate<TDAT>& ts);
	void operator-=(const TensorTemplate<TDAT>& ts);
	void operator*=(const TensorTemplate<TDAT>& ts);
	void operator/=(const TensorTemplate<TDAT>& ts);

	template <typename TNUM>
	void operator=(const TNUM& num);

	template <typename TNUM>
	void operator+=(TNUM num);
	
	template <typename TNUM>
	void operator-=(TNUM num);
	
	template <typename TNUM>
	void operator*=(TNUM num);
	
	template <typename TNUM>
	void operator/=(TNUM num);

	template <typename TNUM>
	operator TensorTemplate<TNUM>() {
		TensorTemplate<TNUM> ts;
		ts.Resize(rows_, cols_, slis_, gros_);
		TNUM* data_ts = ts.data();
		if (numel_ >0) {
			for (int n = 0; n < numel_; ++n) {
				data_ts[n] = data_[n];
			}
		}
		return ts;
	}


protected:
	int rows_;
	int cols_;
	int slis_;
	int gros_;
	int numel_;
	int dims_;
	TDAT* data_;
};


template <typename TDAT>
TensorTemplate<TDAT>::TensorTemplate()
: rows_(0)
, cols_(0)
, slis_(0)
, gros_(0)
, numel_(0)
, dims_(0)
, data_(nullptr) {
	T_CHECK_TYPE(TDAT);
}


template <typename TDAT>
TensorTemplate<TDAT>::TensorTemplate(const TDAT& num)
: rows_(1)
, cols_(1)
, slis_(1)
, gros_(1)
, numel_(1)
, dims_(0)
, data_(nullptr) {
	T_CHECK_TYPE(TDAT);
	data_ = (TDAT*)malloc(sizeof(TDAT));
	*data_ = num;
}


template <typename TDAT>
TensorTemplate<TDAT>::TensorTemplate(const std::vector<TDAT>& list) {
	T_CHECK_TYPE(TDAT);
	rows_ = 1;
	cols_ = list.size();
	slis_ = 1;
	gros_ = 1;
	UPDATE_DIMS_AND_NUMEL();
	size_t length = numel_ * sizeof(TDAT);
	data_ = (TDAT*)malloc(length);
	if (numel_ > 0) {
		for (int j = 0; j < cols_; ++j) {
			data_[j] = list[j];
		}
	}
}


template <typename TDAT>
TensorTemplate<TDAT>::TensorTemplate(const std::vector<std::vector<TDAT> >& array) {
	T_CHECK_TYPE(TDAT);
	rows_ = array.size();
	if (rows_ != 0) {
		cols_ = array[0].size();
	}
	slis_ = 1;
	gros_ = 1;
	UPDATE_DIMS_AND_NUMEL();
	size_t length = numel_ * sizeof(TDAT);
	data_ = (TDAT*)malloc(length);
	if (numel_ > 0) {
		for (int i = 0; i < rows_; ++i) {
			for (int j = 0; j < cols_; ++j) {
				data_[i * cols_ +j] = array[i][j];
			}
		}
	}
}


template <typename TDAT>
TensorTemplate<TDAT>::TensorTemplate(const std::vector<std::vector<std::vector<TDAT> > >& cube) {
	T_CHECK_TYPE(TDAT);
	slis_ = cube.size();
	if (slis_ != 0) {
		rows_ = cube[0].size();
		if (rows_ != 0) {
			cols_ = cube[0][0].size();
		}
	}
	gros_ = 1;
	UPDATE_DIMS_AND_NUMEL();
	size_t length = numel_ * sizeof(TDAT);
	data_ = (TDAT*)malloc(length);
	if (numel_ > 0) {
		for (int k = 0; k < slis_; ++k) {
			int k_base = k * rows_ * cols_;
			for (int i = 0; i < rows_; ++i) {
				for (int j = 0; j < cols_; ++j) {
					data_[k_base + i * cols_ +j] = cube[k][i][j];
				}
			}
		}
	}
}


template <typename TDAT>
TensorTemplate<TDAT>::TensorTemplate(const TensorTemplate<TDAT>& ts) {
	T_CHECK_TYPE(TDAT);
	rows_ = ts.rows();
	cols_ = ts.cols();
	slis_ = ts.slis();
	gros_ = ts.gros();
	UPDATE_DIMS_AND_NUMEL();
	size_t length = numel_ * sizeof(TDAT);
	data_ = (TDAT*)malloc(length);
	TDAT* data_ts = ts.data();
	if (numel_ > 0) {
		memcpy(data_, data_ts, length);
	}
}


template <typename TDAT>
TensorTemplate<TDAT>::TensorTemplate(const std::string& str) {
	T_CHECK_TYPE(TDAT);
	slis_ = 1;
	gros_ = 1;
	std::vector<std::vector<TDAT> > array;
	std::vector<std::string> segments;
	SplitTensorString(str, ";", segments);
	size_t str_cols = 0;
	for (auto i = segments.begin(); i != segments.end(); ++i) {
		std::vector<std::string> particles;
		std::vector<TDAT> particle_values;
		SplitTensorString(*i, ",", particles);
		if (i == segments.begin()) {
			str_cols = particles.size();
		}
		else {
			if (str_cols != particles.size()) {
				T_ERROR("Dimension mismatched.\n");
			}
		}
		for (auto j = particles.begin(); j != particles.end(); ++j) {
			if (typeid(int32_t).name() == typeid(TDAT).name()) {
				particle_values.push_back(atoi(j->c_str()));
			}
			else if (typeid(float).name() == typeid(TDAT).name()) {
				particle_values.push_back(atof(j->c_str()));
			}
		}
		array.push_back(particle_values);
	}
	rows_ = array.size();
	if (rows_ != 0) {
		cols_ = array[0].size();
	}
	UPDATE_DIMS_AND_NUMEL();
	size_t length = numel_ * sizeof(TDAT);
	data_ = (TDAT*)malloc(length);
	if (numel_ > 0) {
		for (int i = 0; i < rows_; ++i) {
			for (int j = 0; j < cols_; ++j) {
				data_[i * cols_ +j] = array[i][j];
			}
		}
	}
}


template <typename TDAT>
TensorTemplate<TDAT>::~TensorTemplate() {
	if (nullptr != data_) {
		free(data_);
		data_ = nullptr;
	}
}


template <typename TDAT>
TDAT TensorTemplate<TDAT>::Item() {
	TDAT num = 0;
	if (1 == numel_) {
		num = data_[0];
		return num;
	}
	else if (0 == numel_) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Cannot convert an empty tensor to a number.");
		#else
		return num;
		#endif
	}
	else {
		#ifdef TENSOR_DEBUG
		T_ERROR("Cannot convert a tensor to a number.");
		#else
		num = data_[0];
		return num;
		#endif
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::Clear() {
	rows_ = 0;
	cols_ = 0;
	slis_ = 0;
	gros_ = 0;
	UPDATE_DIMS_AND_NUMEL();
	if (nullptr != data_) {
		free(data_);
		data_ = nullptr;
	}
}


template <typename TDAT>
int TensorTemplate<TDAT>::Size(int dim) {
	switch (dim) {
		case 0:
		return rows_;
		break;
		case 1:
		return cols_;
		break;
		case 2:
		return slis_;
		break;
		case 3:
		return gros_;
		break;
		default:
		T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::Resize(int rows, int cols, int slis, int gros) {
	if (rows < 0 || cols < 0 || slis < 0 || gros < 0) {
	#ifdef TENSOR_DEBUG
		T_ERROR("The shape of a Tensor should be non-negative.");
	#else
		rows_ = abs(rows);
		cols_ = abs(cols);
		slis_ = abs(slis);
		gros_ = abs(gros);
	#endif
	}
	else {
		rows_ = rows;
		cols_ = cols;
		slis_ = slis;
		gros_ = gros;
	}
	UPDATE_DIMS_AND_NUMEL();
	if (nullptr != data_) {
		free(data_);
		data_ = nullptr;
	}
	data_ = (TDAT*)malloc(numel_ * sizeof(TDAT));
}


template <typename TDAT>
TensorTemplate<TDAT> TensorTemplate<TDAT>::Reshape(int rows, int cols, int slis, int gros) {
	#ifdef TENSOR_DEBUG
	if (rows < 0 || cols < 0 || slis < 0 || gros < 0) {
		T_ERROR("The shape of a Tensor should be non-negative.");
	}
	#else
	rows = abs(rows);
	cols = abs(cols);
	slis = abs(slis);
	gros = abs(gros);
	#endif
	if (rows * cols * slis * gros == numel_) {
		rows_ = rows;
		cols_ = cols;
		slis_ = slis;
		gros_ = gros;
		UPDATE_DIMS_AND_NUMEL();
	}
	else {
		#ifdef TENSOR_DEBUG
		T_ERROR("Cannot change the number of elements when using Reshape.");
		#endif
	}
	return *this;
}


template <typename TDAT>
void TensorTemplate<TDAT>::Transpose() {
	if (0 == numel_) {
		std::swap(rows_, cols_);
	}
	else {
		#ifdef TENSOR_DEBUG
		if (1 != slis_ || 1 != gros_) {
			T_ERROR("Cannot apply Transpose to 3-D or 4-D Tensors.");
		}
		else {
			int rows_ts = rows_;
			int cols_ts = cols_;
			rows_ = cols_ts;
			cols_ = rows_ts;
			TDAT* data_ts = data_;
			size_t length = numel_ * sizeof(TDAT);
			data_ = (TDAT*)malloc(length);
			for (int r = 0; r < rows_; ++r) {
				for (int c = 0; c < cols_; ++c) {
					data_[r * cols_ + c] = data_ts[c * rows_ + r];
				}
			}
			if (nullptr != data_ts) {
				free(data_ts);
				data_ts = nullptr;
			}
		}
		#else
		int rows_ts = rows_;
		int cols_ts = cols_;
		rows_ = cols_ts;
		cols_ = rows_ts;
		TDAT* data_ts = data_;
		size_t length = numel_ * sizeof(TDAT);
		data_ = (TDAT*)malloc(length);
		for (int g = 0; g < gros_; ++g) {
			int g_base = g * slis_ * cols_ * rows_;
			for (int s = 0; s < slis_; ++s) {
				int s_base = s * cols_ * rows_;
				for (int r = 0; r < rows_; ++r) {
					for (int c = 0; c < cols_; ++c) {
						data_[g_base + s_base + r * cols_ + c] = data_ts[g_base + s_base + c * rows_ + r];
					}
				}
			}
		}
		if (nullptr != data_ts) {
			free(data_ts);
			data_ts = nullptr;
		}
		#endif
	}
}


template <typename TDAT>
bool TensorTemplate<TDAT>::IsEmpty() {
	return (0 == numel_);
}


template <typename TDAT>
void TensorTemplate<TDAT>::Zeros() {
	memset(data_, 0, numel_ * sizeof(TDAT));
}


template <typename TDAT>
void TensorTemplate<TDAT>::Ones() {
	auto data = data_;
	for(int i = 0; i < numel_; ++i) {
		*data++ = static_cast<TDAT>(1);
	}
}


template <typename TDAT>
TensorTemplate<TDAT> TensorTemplate<TDAT>::Sum() {
	TensorTemplate<TDAT> ts;
	if (0 != numel_) {
		ts.Resize(1, 1, 1, 1);
		TDAT* data_ts = ts.data();
		for (int n = 0; n < numel_; ++n) {
			data_ts[0] += data_[n];
		}
	}
	return ts;
}


template <typename TDAT>
void TensorTemplate<TDAT>::Info(std::ostream& out) {
	if (IsEmpty()) {
		out << "[ Empty " << rows_ << " * " << cols_ << " * " << slis_ << " * " << gros_ << "  <";
		out << typeid(TDAT).name() << ">  "<< dims_ <<"-D  Tensor ]" << std::endl;
	}
	else {
		out << "[ " << rows_ << " * " << cols_ << " * " << slis_ << " * " << gros_ << "  <";
		out << typeid(TDAT).name() << ">  "<< dims_ <<"-D  Tensor ]" << std::endl;
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::Show(std::ostream& out) {
	if (IsEmpty()) {
		out << "[ Empty " << rows_ << " * " << cols_ << " * " << slis_ << " * " << gros_ << "  <";
		out << typeid(TDAT).name() << ">  "<< dims_ <<"-D  Tensor ]" << std::endl;
	}
	else {
		if (typeid(int32_t).name() == typeid(TDAT).name()) {
			out << "[ " << rows_ << " * " << cols_ << " * " << slis_ << " * " << gros_ << "  <";
			out << typeid(int32_t).name() << ">  "<< dims_ <<"-D  Tensor ]" << std::endl;
			for (int g = 0; g < gros_; ++g) {
				int g_base = g * rows_ * cols_ * slis_;
				if (gros_ > 1) {
					out << "........................................Group #" << g << std::endl;
				}
				for (int s = 0; s < slis_; ++s) {
					int s_base = s * rows_ * cols_;
					if (slis_ > 1) {
						if (gros_ > 1) {
							out << "Group #" << g << ", Slice #" << s << " : " << std::endl;
						}
						else {
							out << "Slice #" << s << " : " << std::endl;
						}
					}
					for (int r = 0; r < rows_; ++r) {
						for (int c = 0; c < cols_; ++c) {
							out << std::setw(6) << data_[g_base + s_base + r * cols_ + c];
						}
						out << std::endl;
					}
					out << std::endl;
				}
				out << std::endl;
			}
		}
		else if (typeid(float).name() == typeid(TDAT).name()) {
			out << "[ " << rows_ << " * " << cols_ << " * " << slis_ << " * " << gros_ << "  <";
			out << typeid(float).name() << ">  "<< dims_ <<"-D  Tensor ]" << std::endl;
			int n = 0;
			float maxV = abs(data_[0]);
			for (int n = 0; n < numel_; ++n) {
				if (abs(data_[n]) > maxV) {
					maxV = data_[n];
				}
			}
			float digits = maxV;
			while(digits >= 1.0) {
				digits /= 10;
				++n;
			}
			if (n == 0) {
				n = 1;
			}
			int pre = 4;
			int wid = n + pre + 3;
			out << std::showpoint;
			if (maxV > 999.99994) {
				out << std::resetiosflags(std::ios::fixed);
				out << std::setiosflags(std::ios::scientific);
			}
			else {
				out << std::resetiosflags(std::ios::fixed);
				out << std::setiosflags(std::ios::fixed);
			}
			out << std::setprecision(pre);
			for (int g = 0; g < gros_; ++g) {
				int g_base = g * rows_ * cols_ * slis_;
				if (gros_ > 1) {
					out << "........................................Group #" << g << std::endl;
				}
				for (int s = 0; s < slis_; ++s) {
					int s_base = s * rows_ * cols_;
					if (slis_ > 1) {
						if (gros_ > 1) {
							out << "Group #" << g << ", Slice #" << s << " : " << std::endl;
						}
						else {
							out << "Slice #" << s << " : " << std::endl;
						}
					}
					for (int r = 0; r < rows_; ++r) {
						for (int c = 0; c < cols_; ++c) {
							out << std::setw(wid) << data_[g_base + s_base + r * cols_ + c];
						}
						out << std::endl;
					}
					out << std::endl;
				}
				out << std::endl;
			}
			out << std::setprecision(6);
			out << std::noshowpoint;
		}
	}
}


template <typename TDAT>
TensorTemplate<TDAT> TensorTemplate<TDAT>::S(int ra, int rb, int ca, int cb, int sa, int sb, int ga, int gb) {
	// Set default values
	if (ra == -1) ra = 0;
	if (rb == -1) rb = rows_;
	if (ca == -1) ca = 0;
	if (cb == -1) cb = cols_;
	if (sa == -1) sa = 0;
	if (sb == -1) sb = slis_;
	if (ga == -1) ga = 0;
	if (gb == -1) gb = gros_;

	// Swap if necessary
	if (ra > rb) std::swap(ra, rb);
	if (ca > cb) std::swap(ca, cb);
	if (sa > sb) std::swap(sa, sb);
	if (ga > gb) std::swap(ga, gb);

	// If out of range
	if (ra < 0 || rb > rows_ || ca < 0 || cb > cols_ || sa < 0 || sa > slis_ || ga < 0 || gb > gros_) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Index out of range.");
		#endif
		TensorTemplate<TDAT> ts;
		return ts;
	}

	else {
		// Prepare an empty Tensor for output
		int rows_ts = rb - ra;
		int cols_ts = cb - ca;
		int slis_ts = sb - sa;
		int gros_ts = gb - ga;
		TensorTemplate<TDAT> ts;
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		TDAT* data_ts = ts.data();

		// Copy data
		for (int g = 0; g < gros_ts; ++g) {
			int g_base_ts = g * rows_ts * cols_ts * slis_ts;
			int g_base = (g + ga) * rows_ * cols_ * slis_;
			for (int s = 0; s < slis_ts; ++s) {
				int s_base_ts = s * rows_ts * cols_ts;
				int s_base = (s + sa) * rows_ * cols_;
				for (int r = 0; r < rows_ts; ++r) {
					for (int c = 0; c < cols_ts; ++c) {
						data_ts[g_base_ts + s_base_ts + r * cols_ts + c] = data_[g_base + s_base + (r + ra) * cols_ + c + ca];
					}
				}
			}
		}
		return ts;
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::S(const TensorTemplate<TDAT>& a, int ra, int rb, int ca, int cb, int sa, int sb, int ga, int gb) {
	// Set default values
	if (ra == -1) ra = 0;
	if (rb == -1) rb = rows_;
	if (ca == -1) ca = 0;
	if (cb == -1) cb = cols_;
	if (sa == -1) sa = 0;
	if (sb == -1) sb = slis_;
	if (ga == -1) ga = 0;
	if (gb == -1) gb = gros_;

	// Swap if necessary
	if (ra > rb) std::swap(ra, rb);
	if (ca > cb) std::swap(ca, cb);
	if (sa > sb) std::swap(sa, sb);
	if (ga > gb) std::swap(ga, gb);

	// If out of range
	if (ra < 0 || rb > rows_ || ca < 0 || cb > cols_ || sa < 0 || sa > slis_ || ga < 0 || gb > gros_) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Index out of range.\n");
		#else
		T_WARNING("Index out of range.\n");
		#endif
	}

	else {
		int rows_ts = rb - ra;
		int cols_ts = cb - ca;
		int slis_ts = sb - sa;
		int gros_ts = gb - ga;
		int rows_a = a.rows();
		int cols_a = a.cols();
		int slis_a = a.slis();
		int gros_a = a.gros();
		int numel_a = a.numel();
		if (1 == numel_a) {
			TDAT* data_a = a.data();
			TDAT temp = data_a[0];
			for (int g = 0; g < gros_a; ++g) {
				int g_base = (g + ga) * rows_ * cols_ * slis_;
				for (int s = 0; s < slis_a; ++s) {
					int s_base = (s + sa) * rows_ * cols_;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_[g_base + s_base + (r + ra) * cols_ + c + ca] = temp;
						}
					}
				}
			}
		}
		else if (rows_a != rows_ts || cols_a != cols_ts || slis_a != slis_ts || gros_a != gros_ts) {
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension mismatched.\n");
			#else
			T_WARNING("Dimension mismatched.\n");
			#endif
		}
		else {
			TDAT* data_a = a.data();

			// Copy data
			for (int g = 0; g < gros_a; ++g) {
				int g_base_a = g * rows_a * cols_a * slis_a;
				int g_base = (g + ga) * rows_ * cols_ * slis_;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_a = s * rows_a * cols_a;
					int s_base = (s + sa) * rows_ * cols_;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_[g_base + s_base + (r + ra) * cols_ + c + ca] = data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
		}
	}
}


template <typename TDAT>
TDAT& TensorTemplate<TDAT>::operator()(int r, int c, int s, int g) {
	if (0 == numel_) {
		T_ERROR("Index out of range. The tensor is empty.");
	}
	else {
		if (1 == dims_ && numel_ > 1 && r > 0 && 0 == c && 0 == s && 0 == g) {
			if (r < numel_)
				return data_[r];
			else {
				#ifdef TENSOR_DEBUG
				T_ERROR("Index out of range.");
				#else
				T_WARNING("Index out of range.");
				return data_[0];
				#endif
			}
		}
		else if (r >= rows_ || c >= cols_ || s >= slis_ || g >= gros_ || r < 0 || c < 0 || s < 0 || g < 0) {
			#ifdef TENSOR_DEBUG
			T_ERROR("Index out of range.");
			#else
			return data_[0];
			#endif
		}
		else {
			return (data_[g * rows_ * cols_ * slis_ + s * rows_ * cols_ + r * cols_ + c]);
		}
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::operator=(const TensorTemplate<TDAT>& ts) {
	int rows_ts = ts.rows();
	int cols_ts = ts.cols();
	int slis_ts = ts.slis();
	int gros_ts = ts.gros();
	TDAT* data_ts = ts.data();
	int numel_ts = ts.numel();
	size_t length = numel_ts * sizeof(TDAT);
	if (rows_ != rows_ts || cols_ != cols_ts || slis_ != slis_ts || gros_ != gros_ts) {
		rows_ = rows_ts;
		cols_ = cols_ts;
		slis_ = slis_ts;
		gros_ = gros_ts;
		UPDATE_DIMS_AND_NUMEL();
		if (nullptr != data_) {
			free(data_);
			data_ = nullptr;
		}
		data_ = (TDAT*)malloc(length);
	}
	memcpy(data_, data_ts, length);
}


template <typename TDAT>
void TensorTemplate<TDAT>::operator+=(const TensorTemplate<TDAT>& ts) {
	int rows_ts = ts.rows();
	int cols_ts = ts.cols();
	int slis_ts = ts.slis();
	int gros_ts = ts.gros();
	TDAT* data_ts = ts.data();
	int numel_ts = ts.numel();
	if (rows_ == rows_ts && cols_ == cols_ts && slis_ == slis_ts && gros_ == gros_ts) {
		for (int n = 0; n < numel_ts; ++n) {
			data_[n] += data_ts[n];
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::operator-=(const TensorTemplate<TDAT>& ts) {
	int rows_ts = ts.rows();
	int cols_ts = ts.cols();
	int slis_ts = ts.slis();
	int gros_ts = ts.gros();
	TDAT* data_ts = ts.data();
	int numel_ts = ts.numel();
	if (rows_ == rows_ts && cols_ == cols_ts && slis_ == slis_ts && gros_ == gros_ts) {
		for (int n = 0; n < numel_ts; ++n) {
			data_[n] -= data_ts[n];
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::operator*=(const TensorTemplate<TDAT>& ts) {
	int rows_ts = ts.rows();
	int cols_ts = ts.cols();
	int slis_ts = ts.slis();
	int gros_ts = ts.gros();
	TDAT* data_ts = ts.data();
	int numel_ts = ts.numel();
	if (rows_ == rows_ts && cols_ == cols_ts && slis_ == slis_ts && gros_ == gros_ts) {
		for (int n = 0; n < numel_ts; ++n) {
			data_[n] *= data_ts[n];
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


template <typename TDAT>
void TensorTemplate<TDAT>::operator/=(const TensorTemplate<TDAT>& ts) {
	int rows_ts = ts.rows();
	int cols_ts = ts.cols();
	int slis_ts = ts.slis();
	int gros_ts = ts.gros();
	TDAT* data_ts = ts.data();
	int numel_ts = ts.numel();
	if (rows_ == rows_ts && cols_ == cols_ts && slis_ == slis_ts && gros_ == gros_ts) {
		for (int n = 0; n < numel_ts; ++n) {
			data_[n] /= data_ts[n];
		}
	}
	else {
		T_ERROR("Dimension mismatched.\n");
	}
}


template <typename TDAT>
template <typename TNUM>
void TensorTemplate<TDAT>::operator=(const TNUM& num) {
	rows_ = 1;
	cols_ = 1;
	slis_ = 1;
	gros_ = 1;
	UPDATE_DIMS_AND_NUMEL();
	if (nullptr != data_) {
		free(data_);
		data_ = nullptr;
	}
	data_ = (TDAT*)malloc(sizeof(TDAT));
	*data_ = num;
}


template <typename TDAT>
template <typename TNUM>
void TensorTemplate<TDAT>::operator+=(TNUM num) {
	for (int i = 0; i < numel_; ++i)
		data_[i] += num;
}


template <typename TDAT>
template <typename TNUM>
void TensorTemplate<TDAT>::operator-=(TNUM num) {
	for (int i = 0; i < numel_; ++i)
		data_[i] -= num;
}


template <typename TDAT>
template <typename TNUM>
void TensorTemplate<TDAT>::operator*=(TNUM num) {
	if (0 == num) {
		Zeros();
	}
	else if (1 == num) {
	}
	else {
		for (int i = 0; i < numel_; ++i) {
			data_[i] *= num;
		}
	}
}


template <typename TDAT>
template <typename TNUM>
void TensorTemplate<TDAT>::operator/=(TNUM num) {
	if (1 == num) {
	}
	else {
		if (0 == num) {
			T_ERROR("Division by zero.");
		}
		for (int i = 0; i < numel_; ++i) {
			data_[i] /= num;
		}
	}
}


// ****************************************
// Static functions
// ****************************************
template <typename TDAT>
inline TensorTemplate<TDAT> RawTemplate(int rows, int cols, int slis, int gros) {
	TensorTemplate<TDAT> ts;
	ts.Resize(rows, cols, slis, gros);
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> RawTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	ts.Resize(a.rows(), a.cols(), a.slis(), a.gros());
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> ZerosTemplate(int rows, int cols, int slis, int gros) {
	TensorTemplate<TDAT> ts;
	ts.Resize(rows, cols, slis, gros);
	ts.Zeros();
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> ZerosTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	ts.Resize(a.rows(), a.cols(), a.slis(), a.gros());
	ts.Zeros();
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> OnesTemplate(int rows, int cols, int slis, int gros) {
	TensorTemplate<TDAT> ts;
	ts.Resize(rows, cols, slis, gros);
	ts.Ones();
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> OnesTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	ts.Resize(a.rows(), a.cols(), a.slis(), a.gros());
	ts.Ones();
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> ArangeTemplate(int num) {
	TensorTemplate<TDAT> ts;
	if (0 > num) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Number of elements should be non-negative.\n");
		#else
		ts.Resize(1, 0, 1, 1);
		return ts;
		#endif
	}
	else if (0 == num) {
		ts.Resize(1, 0, 1, 1);
		return ts;
	}
	else {
		ts.Resize(1, num, 1, 1);
		TDAT* data_ts = ts.data();
		for (int n = 0; n < num; ++n) {
			*data_ts++ = n;
		}
		return ts;
	}
}


template <typename TDAT>
inline bool MatchTemplate(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& b) {
	return (a.rows() == b.rows() && a.cols() == b.cols() && a.slis() == b.slis() && a.gros() == b.gros());
}


template <typename TDAT>
inline int NumelTemplate(const TensorTemplate<TDAT>& a) {
	return a.numel();
}


template <typename TDAT>
inline TensorTemplate<int32_t> SizeTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<int32_t> ts;
	ts.Resize(1, 4, 1, 1);
	int32_t* data_ts = ts.data();
	data_ts[0] = a.rows();
	data_ts[1] = a.cols();
	data_ts[2] = a.slis();
	data_ts[3] = a.gros();
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> ReshapeTemplate(const TensorTemplate<TDAT>& a, int rows, int cols, int slis, int gros) {
	TensorTemplate<TDAT> ts(a);
	return ts.Reshape(rows, cols, slis, gros);
}


template <typename TDAT>
inline TensorTemplate<TDAT> TransposeTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		ts.Resize(cols_a, rows_a, slis_a, gros_a);
	}
	else {
		#ifdef TENSOR_DEBUG
		if (1 != slis_a || 1 != gros_a) {
			T_ERROR("Cannot apply Transpose to 3-D or 4-D Tensors.");
		}
		else {
			ts.Resize(cols_a, rows_a, slis_a, gros_a);
			TDAT* data_ts = ts.data();
			TDAT* data_a = a.data();
			for (int r = 0; r < rows_a; ++r) {
				for (int c = 0; c < cols_a; ++c) {
					data_ts[c * rows_a + r] = data_a[r * cols_a + c];
				}
			}
		}
		#else
		ts.Resize(cols_a, rows_a, slis_a, gros_a);
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();
		for (int g = 0; g < gros_a; ++g) {
			int g_base = g * slis_a * cols_a * rows_a;
			for (int s = 0; s < slis_a; ++s) {
				int s_base = s * cols_a * rows_a;
				for (int r = 0; r < rows_a; ++r) {
					for (int c = 0; c < cols_a; ++c) {
						data_ts[g_base + s_base + c * rows_a + r] = *data_a++;
					}
				}
			}
		}
		#endif
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> FlipTemplate(const TensorTemplate<TDAT>& a, int dim) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (dim < 0 || dim > 3) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
		return a;
		#else
		return a;
		#endif
	}
	ts.Resize(rows_a, cols_a, slis_a, gros_a);
	if (0 != numel_a) {
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();

		// 0th dimension
		if (0 == dim) {
			for (int g = 0; g < gros_a; ++g) {
				int g_base = g * slis_a * cols_a * rows_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base = s * cols_a * rows_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base + s_base + (rows_a - r - 1) * cols_a + c] = data_a[g_base + s_base + r * cols_a + c];
						}
					}
				}
			}
		}

		// 1st dimension
		if (1 == dim) {
			for (int g = 0; g < gros_a; ++g) {
				int g_base = g * slis_a * cols_a * rows_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base = s * cols_a * rows_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base + s_base + r * cols_a + (cols_a - c - 1)] = data_a[g_base + s_base + r * cols_a + c];
						}
					}
				}
			}
		}

		// 2nd dimension
		if (2 == dim) {
			for (int g = 0; g < gros_a; ++g) {
				int g_base = g * slis_a * cols_a * rows_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base = s * cols_a * rows_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base + (slis_a - s - 1) * cols_a * rows_a + r * cols_a + c] = data_a[g_base + s_base + r * cols_a + c];
						}
					}
				}
			}
		}

		// 3rd dimension
		if (3 == dim) {
			for (int g = 0; g < gros_a; ++g) {
				int g_base = g * slis_a * cols_a * rows_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base = s * cols_a * rows_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[(gros_a - g - 1) * slis_a * cols_a * rows_a + s_base + r * cols_a + c] = data_a[g_base + s_base + r * cols_a + c];
						}
					}
				}
			}
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> FlipTemplate(const TensorTemplate<TDAT>& a, const std::string& dim_string) {
	int dim = 0;
	if ("ud" == dim_string || "r" == dim_string) {
		dim = 0;
	}
	else if ("lr" == dim_string || "c" == dim_string) {
		dim = 1;
	}
	else if ("bf" == dim_string || "s" == dim_string) {
		dim = 2;
	}
	else if ("g" == dim_string) {
		dim = 3;
	}
	else {
		dim = -1;
	}
	return FlipTemplate(a, dim);
}


template <typename TDAT>
inline TensorTemplate<TDAT> RepmatTemplate(const TensorTemplate<TDAT>& a, int rt, int ct, int st, int gt) {
	if (rt < 1 || ct < 1 || st < 1 || gt < 1) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Multiples should be positive integers.\n");
		#else
		T_WARNING("Multiples should be positive integers.\n");
		#endif
		TensorTemplate<TDAT> ts(a);
		return ts;
	}
	else {
		TensorTemplate<TDAT> ts;
		int numel_a = a.numel();
		int rows_a = a.rows();
		int cols_a = a.cols();
		int slis_a = a.slis();
		int gros_a = a.gros();
		int rows_ts = rows_a * rt;
		int cols_ts = cols_a * ct;
		int slis_ts = slis_a * st;
		int gros_ts = gros_a * gt;
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		if (numel_a > 0) {
			TDAT* data_a = a.data();
			TDAT* data_ts = ts.data();
			int ar = 0, ac = 0, as = 0, ag = 0;
			for (int g = 0; g < gros_ts; ++g) {
				int g_base_ts = g * rows_ts * cols_ts * slis_ts;
				int g_base_a = ag * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_ts; ++s) {
					int s_base_ts = s * rows_ts * cols_ts;
					int s_base_a = as * rows_a * cols_a;
					for (int r = 0; r < rows_ts; ++r) {
						for (int c = 0; c < cols_ts; ++c) {
							data_ts[g_base_ts + s_base_ts + r * cols_ts + c] = data_a[g_base_a + s_base_a + ar * cols_a + ac];
							if (++ac == cols_a) ac = 0;
						}
						if (++ar == rows_a) ar = 0;
					}
					if (++as == slis_a) as = 0;
				}
				if (++ag == gros_a) ag = 0;
			}
		}
		return ts;
	}
}


template <typename TDAT>
inline TensorTemplate<TDAT> KronTemplate(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& b) {
	int rt = b.rows();
	int ct = b.cols();
	int st = b.slis();
	int gt = b.gros();
	if (rt < 1 || ct < 1 || st < 1 || gt < 1) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Multiples should be positive integers.\n");
		#else
		T_WARNING("Multiples should be positive integers.\n");
		#endif
		TensorTemplate<TDAT> ts(a);
		return ts;
	}
	else {
		TensorTemplate<TDAT> ts;
		int numel_a = a.numel();
		int rows_a = a.rows();
		int cols_a = a.cols();
		int slis_a = a.slis();
		int gros_a = a.gros();
		int rows_ts = rows_a * rt;
		int cols_ts = cols_a * ct;
		int slis_ts = slis_a * st;
		int gros_ts = gros_a * gt;
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		if (numel_a > 0) {
			TDAT* data_a = a.data();
			TDAT* data_b = b.data();
			TDAT* data_ts = ts.data();
			int ar = 0, ac = 0, as = 0, ag = 0;
			int br = 0, bc = 0, bs = 0, bg = 0;
			for (int g = 0; g < gros_ts; ++g) {
				ag = g / gt;
				bg = g % gt;
				int g_base_a = ag * rows_a * cols_a * slis_a;
				int g_base_b = bg * rt * ct * st;
				for (int s = 0; s < slis_ts; ++s) {
					as = s / st;
					bs = s % st;
					int s_base_a = as * rows_a * cols_a;
					int s_base_b = bs * rt * ct;
					for (int r = 0; r < rows_ts; ++r) {
						ar = r / rt;
						br = r % rt;
						for (int c = 0; c < cols_ts; ++c) {
							ac = c / ct;
							bc = c % ct;
							*data_ts++ = data_a[g_base_a + s_base_a + ar * cols_a + ac] * data_b[g_base_b + s_base_b + br * ct + bc];
						}
					}
				}
			}
		}
		return ts;
	}
}


template <typename TDAT>
inline TensorTemplate<TDAT> PermuteTemplate(const TensorTemplate<TDAT>& a, int dim0, int dim1, int dim2, int dim3) {
	int dim[4] = {0, 0, 0, 0};
	dim[0] = dim0;
	dim[1] = dim1;
	dim[2] = dim2;
	dim[3] = dim3;
	bool valid = false;
	for (int d = 0; d < 4; ++d) {
		valid = false;
		for (int t = 0; t < 4; ++t) {
			if (d == dim[t]) {
				valid = true;
				break;
			}
		}
		if (!valid) {
			break;
		}
	}
	if (valid) {
		TensorTemplate<TDAT> ts;
		int scale_a[4];
		int x[4];
		int rearrange[4] = {1, 0, 2, 3};
		int rows_a = a.rows();
		int cols_a = a.cols();
		int slis_a = a.slis();
		int gros_a = a.gros();
		scale_a[0] = rows_a;
		scale_a[1] = cols_a;
		scale_a[2] = slis_a;
		scale_a[3] = gros_a;
		int rows_ts = scale_a[dim[0]];
		int cols_ts = scale_a[dim[1]];
		int slis_ts = scale_a[dim[2]];
		int gros_ts = scale_a[dim[3]];
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		int numel_a = a.numel();
		for (int d = 0; d < 4; ++d) {
			x[d] = 1;
			int i = 0;
			for (int t = 0; t < 4; ++t) {
				if (i == dim[d]) {
					break;
				}
				if (t == rearrange[d]) {
					continue;
				}
				else {
					x[d] *= scale_a[rearrange[t]];
					i++;
				}
			}
		}
		int x0 = x[0];
		int x1 = x[1];
		int x2 = x[2];
		int x3 = x[3];
		if (0 != numel_a) {
			TDAT* data_a = a.data();
			TDAT* data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * x3;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * x2;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + r * x1 + c * x0] = *data_a++;
						}
					}
				}
			}
		}
		return ts;
	}
	else {
		#ifdef TENSOR_DEBUG
		T_ERROR("Dimensions for Permute should be a rearrangement of [0, 1, 2, 3].\n");
		#else
		T_WARNING("Dimensions for Permute should be a rearrangement of [0, 1, 2, 3].\n");
		#endif
		TensorTemplate<TDAT> ts(a);
		return ts;
	}
}


template <typename TDAT>
inline TensorTemplate<TDAT> Rot90Template(const TensorTemplate<TDAT>& a, int times, int axis) {
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	TensorTemplate<TDAT> ts;
	int position = times % 4;
	if (axis < 0 || axis > 3) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
		return a;
		#else
		return a;
		#endif
	}
	else {
		int rows_ts;
		int cols_ts;
		int slis_ts;
		int gros_ts;
		if (0 == axis) {
			switch (position) {
				case 1:
				case -3:
				rows_ts = rows_a;
				cols_ts = slis_a;
				slis_ts = cols_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int c = 0; c < cols_ts; ++c) {
							for (int r = 0; r < rows_ts; ++r) {
								for (int s = 0; s < slis_ts; ++s) {
									data_ts[g * rows_ts * cols_ts * slis_ts + (slis_ts - s - 1) * rows_ts * cols_ts + r * cols_ts + c] = *data_a++;
								}
							}
						}
					}
				}
				break;
				case 2:
				case -2:
				rows_ts = rows_a;
				cols_ts = cols_a;
				slis_ts = slis_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int s = 0; s < slis_ts; ++s) {
							for (int r = 0; r < rows_ts; ++r) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g * rows_ts * cols_ts * slis_ts + (slis_ts - s - 1) * rows_ts * cols_ts + r * cols_ts + (cols_ts - c - 1)] = *data_a++;
								}
							}
						}
					}
				}
				break;
				case 3:
				case -1:
				rows_ts = rows_a;
				cols_ts = slis_a;
				slis_ts = cols_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int c = 0; c < cols_ts; ++c) {
							for (int r = 0; r < rows_ts; ++r) {
								for (int s = 0; s < slis_ts; ++s) {
									data_ts[g * rows_ts * cols_ts * slis_ts + s * rows_ts * cols_ts + r * cols_ts + (cols_ts - c - 1)] = *data_a++;
								}
							}
						}
					}
				}
				break;
				default:
				ts = a;
			}
		}
		else if (1 == axis) {
			switch (position) {
				case 1:
				case -3:
				rows_ts = slis_a;
				cols_ts = cols_a;
				slis_ts = rows_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int r = 0; r < rows_ts; ++r) {
							for (int s = 0; s < slis_ts; ++s) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g * rows_ts * cols_ts * slis_ts + s * rows_ts * cols_ts + (rows_ts - r - 1) * cols_ts + c] = *data_a++;
								}
							}
						}
					}
				}
				break;
				case 2:
				case -2:
				rows_ts = rows_a;
				cols_ts = cols_a;
				slis_ts = slis_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int s = 0; s < slis_ts; ++s) {
							for (int r = 0; r < rows_ts; ++r) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g * rows_ts * cols_ts * slis_ts + (slis_ts - s - 1) * rows_ts * cols_ts + (rows_ts - r - 1) * cols_ts + c] = *data_a++;
								}
							}
						}
					}
				}
				break;
				case 3:
				case -1:
				rows_ts = slis_a;
				cols_ts = cols_a;
				slis_ts = rows_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int r = 0; r < rows_ts; ++r) {
							for (int s = 0; s < slis_ts; ++s) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g * rows_ts * cols_ts * slis_ts + (slis_ts - s - 1) * rows_ts * cols_ts + r * cols_ts + c] = *data_a++;
								}
							}
						}
					}
				}
				break;
				default:
				ts = a;
			}
		}
		else {
			switch (position) {
				case 1:
				case -3:
				rows_ts = cols_a;
				cols_ts = rows_a;
				slis_ts = slis_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int s = 0; s < slis_ts; ++s) {
							for (int c = 0; c < cols_ts; ++c) {
								for (int r = 0; r < rows_ts; ++r) {
									data_ts[g * rows_ts * cols_ts * slis_ts + s * rows_ts * cols_ts + r * cols_ts + (cols_ts - c - 1)] = *data_a++;
								}
							}
						}
					}
				}
				break;
				case 2:
				case -2:
				rows_ts = rows_a;
				cols_ts = cols_a;
				slis_ts = slis_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int s = 0; s < slis_ts; ++s) {
							for (int r = 0; r < rows_ts; ++r) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g * rows_ts * cols_ts * slis_ts + s * rows_ts * cols_ts + (rows_ts - r - 1) * cols_ts + (cols_ts - c - 1)] = *data_a++;
								}
							}
						}
					}
				}
				break;
				case 3:
				case -1:
				rows_ts = cols_a;
				cols_ts = rows_a;
				slis_ts = slis_a;
				gros_ts = gros_a;
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				if (numel_a > 0) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_ts; ++g) {
						for (int s = 0; s < slis_ts; ++s) {
							for (int c = 0; c < cols_ts; ++c) {
								for (int r = 0; r < rows_ts; ++r) {
									data_ts[g * rows_ts * cols_ts * slis_ts + s * rows_ts * cols_ts + (rows_ts - r - 1) * cols_ts + c] = *data_a++;
								}
							}
						}
					}
				}
				break;
				default:
				ts = a;
			}
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> RearrangeTemplate(const TensorTemplate<TDAT>& a, const std::vector<int>& v, int dim) {
	TensorTemplate<TDAT> ts;
	if (dim > 3 || dim < 0) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Dimension for rearranging should be 0, 1, 2, or 3.\n");
		#else
		T_WARNING("Dimension for rearranging should be 0, 1, 2, or 3.\n");
		#endif
		return a;
	}
	else {
		int rows_a = a.rows();
		int cols_a = a.cols();
		int slis_a = a.slis();
		int gros_a = a.gros();
		int numel_a = a.numel();
		int numel_v = v.size();
		ts.Resize(rows_a, cols_a, slis_a, gros_a);
		if (numel_a > 0) {
			if (0 == dim) {
				if (numel_v == rows_a) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_a; ++g) {
						int g_base_ts = g * rows_a * cols_a * slis_a;
						for (int s = 0; s < slis_a; ++s) {
							int s_base_ts = s * rows_a * cols_a;
							for (int r = 0; r < rows_a; ++r) {
								for (int c = 0; c < cols_a; ++c) {
									data_ts[g_base_ts + s_base_ts + v[r] * cols_a + c] = *data_a++;
								}
							}
						}
					}
				}
				else {
					#ifdef TENSOR_DEBUG
					T_ERROR("Dimension mismatched.\n");
					#else
					T_WARNING("Dimension mismatched.\n");
					#endif
					return a;
				}
			}
			else if (1 == dim) {
				if (numel_v == cols_a) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_a; ++g) {
						int g_base_ts = g * rows_a * cols_a * slis_a;
						for (int s = 0; s < slis_a; ++s) {
							int s_base_ts = s * rows_a * cols_a;
							for (int r = 0; r < rows_a; ++r) {
								for (int c = 0; c < cols_a; ++c) {
									data_ts[g_base_ts + s_base_ts + r * cols_a + v[c]] = *data_a++;
								}
							}
						}
					}
				}
				else {
					#ifdef TENSOR_DEBUG
					T_ERROR("Dimension mismatched.\n");
					#else
					T_WARNING("Dimension mismatched.\n");
					#endif
					return a;
				}
			}
			else if (2 == dim) {
				if (numel_v == slis_a) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_a; ++g) {
						int g_base_ts = g * rows_a * cols_a * slis_a;
						for (int s = 0; s < slis_a; ++s) {
							int s_base_ts = v[s] * rows_a * cols_a;
							for (int r = 0; r < rows_a; ++r) {
								for (int c = 0; c < cols_a; ++c) {
									data_ts[g_base_ts + s_base_ts + r * cols_a + c] = *data_a++;
								}
							}
						}
					}
				}
				else {
					#ifdef TENSOR_DEBUG
					T_ERROR("Dimension mismatched.\n");
					#else
					T_WARNING("Dimension mismatched.\n");
					#endif
					return a;
				}
			}
			else {
				if (numel_v == gros_a) {
					TDAT* data_a = a.data();
					TDAT* data_ts = ts.data();
					for (int g = 0; g < gros_a; ++g) {
						int g_base_ts = v[g] * rows_a * cols_a * slis_a;
						for (int s = 0; s < slis_a; ++s) {
							int s_base_ts = s * rows_a * cols_a;
							for (int r = 0; r < rows_a; ++r) {
								for (int c = 0; c < cols_a; ++c) {
									data_ts[g_base_ts + s_base_ts + r * cols_a + c] = *data_a++;
								}
							}
						}
					}
				}
				else {
					#ifdef TENSOR_DEBUG
					T_ERROR("Dimension mismatched.\n");
					#else
					T_WARNING("Dimension mismatched.\n");
					#endif
					return a;
				}
			}
		}
		return ts;
	}
}


template <typename TDAT>
inline TensorTemplate<TDAT> SumTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	ts.Resize(1, 1, 1, 1);
	ts.Zeros();
	int numel_a = a.numel();
	if (0 != numel_a) {
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();
		for (int n = 0; n < numel_a; ++n) {
			data_ts[0] += data_a[n];
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> SumTemplate(const TensorTemplate<TDAT>& a, int dim) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		switch (dim) {
			case 0:
			ts.Resize((0 == rows_a)? 0: 1, cols_a, slis_a, gros_a);
			break;
			case 1:
			ts.Resize(rows_a, (0 == cols_a)? 0: 1, slis_a, gros_a);
			break;
			case 2:
			ts.Resize(rows_a, cols_a, (0 == slis_a)? 0: 1, gros_a);
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, (0 == gros_a)? 0: 1);
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			#endif
		}
	}
	if (0 != numel_a) {
		TDAT* data_ts;
		TDAT* data_a;
		switch (dim) {
			case 0:
			ts.Resize(1, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			break;
			case 1:
			ts.Resize(rows_a, 1, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + r * 1 + 0] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			break;
			case 2:
			ts.Resize(rows_a, cols_a, 1, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, 1);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int n = 0; n < numel_a; ++n) {
				data_ts[n] = data_a[n];
			}
			#endif
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MeanTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	ts.Resize(1, 1, 1, 1);
	ts.Zeros();
	int numel_a = a.numel();
	if (0 != numel_a) {
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();
		for (int n = 0; n < numel_a; ++n) {
			data_ts[0] += data_a[n];
		}
		data_ts[0] /= numel_a;
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MeanTemplate(const TensorTemplate<TDAT>& a, int dim) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		switch (dim) {
			case 0:
			ts.Resize((0 == rows_a)? 0: 1, cols_a, slis_a, gros_a);
			break;
			case 1:
			ts.Resize(rows_a, (0 == cols_a)? 0: 1, slis_a, gros_a);
			break;
			case 2:
			ts.Resize(rows_a, cols_a, (0 == slis_a)? 0: 1, gros_a);
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, (0 == gros_a)? 0: 1);
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			#endif
		}
	}
	if (0 != numel_a) {
		TDAT* data_ts;
		TDAT* data_a;
		switch (dim) {
			case 0:
			ts.Resize(1, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			ts /= rows_a;
			break;
			case 1:
			ts.Resize(rows_a, 1, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + r * 1 + 0] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			ts /= cols_a;
			break;
			case 2:
			ts.Resize(rows_a, cols_a, 1, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			ts /= slis_a;
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, 1);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			ts /= gros_a;
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int n = 0; n < numel_a; ++n) {
				data_ts[n] = data_a[n];
			}
			#endif
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> StddevTemplate(const TensorTemplate<TDAT>& a, const std::string& ddof) {
	TensorTemplate<TDAT> ts;
	ts.Resize(1, 1, 1, 1);
	ts.Zeros();
	int numel_a = a.numel();
	if (0 != numel_a) {
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();
		TDAT mean_value = 0;
		TDAT variance = 0;
		for (int n = 0; n < numel_a; ++n) {
			mean_value += data_a[n];
		}
		mean_value /= numel_a;
		for (int n = 0; n < numel_a; ++n) {
			variance += ((data_a[n] - mean_value) * (data_a[n] - mean_value));
		}
		if ("0" == ddof || 1 == numel_a) {
			data_ts[0] = sqrt(variance / numel_a);
		}
		else {
			data_ts[0] = sqrt(variance / (numel_a - 1));
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> StddevTemplate(const TensorTemplate<TDAT>& a, int dim, const std::string& ddof) {
	TensorTemplate<TDAT> ts;
	TensorTemplate<TDAT> m;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		switch (dim) {
			case 0:
			ts.Resize((0 == rows_a)? 0: 1, cols_a, slis_a, gros_a);
			break;
			case 1:
			ts.Resize(rows_a, (0 == cols_a)? 0: 1, slis_a, gros_a);
			break;
			case 2:
			ts.Resize(rows_a, cols_a, (0 == slis_a)? 0: 1, gros_a);
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, (0 == gros_a)? 0: 1);
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			#endif
		}
	}
	if (0 != numel_a) {
		TDAT* data_ts;
		TDAT* data_a;
		TDAT* data_m;
		int numel_ts;
		int num;
		switch (dim) {
			case 0:
			m.Resize(1, cols_a, slis_a, gros_a);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + 0 * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= rows_a;
			ts.Resize(1, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + 0 * cols_a + c];
							data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == rows_a)? rows_a: (rows_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = sqrt(data_ts[n] / num);
			}
			break;
			case 1:
			m.Resize(rows_a, 1, slis_a, gros_a);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + r * 1 + 0] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= cols_a;
			ts.Resize(rows_a, 1, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + r * 1 + 0];
							data_ts[g_base_ts + s_base_ts + r * 1 + 0] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == cols_a)? cols_a: (cols_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = sqrt(data_ts[n] / num);
			}
			break;
			case 2:
			m.Resize(rows_a, cols_a, 1, gros_a);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= slis_a;
			ts.Resize(rows_a, cols_a, 1, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + r * cols_a + c];
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == slis_a)? slis_a: (slis_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = sqrt(data_ts[n] / num);
			}
			break;
			case 3:
			m.Resize(rows_a, cols_a, slis_a, 1);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= gros_a;
			ts.Resize(rows_a, cols_a, slis_a, 1);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + r * cols_a + c];
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == gros_a)? gros_a: (gros_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = sqrt(data_ts[n] / num);
			}
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int n = 0; n < numel_a; ++n) {
				data_ts[n] = data_a[n];
			}
			#endif
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> VarTemplate(const TensorTemplate<TDAT>& a, const std::string& ddof) {
	TensorTemplate<TDAT> ts;
	ts.Resize(1, 1, 1, 1);
	ts.Zeros();
	int numel_a = a.numel();
	if (0 != numel_a) {
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();
		TDAT mean_value = 0;
		TDAT variance = 0;
		for (int n = 0; n < numel_a; ++n) {
			mean_value += data_a[n];
		}
		mean_value /= numel_a;
		for (int n = 0; n < numel_a; ++n) {
			variance += ((data_a[n] - mean_value) * (data_a[n] - mean_value));
		}
		if ("0" == ddof || 1 == numel_a) {
			data_ts[0] = (variance / numel_a);
		}
		else {
			data_ts[0] = (variance / (numel_a - 1));
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> VarTemplate(const TensorTemplate<TDAT>& a, int dim, const std::string& ddof) {
	TensorTemplate<TDAT> ts;
	TensorTemplate<TDAT> m;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		switch (dim) {
			case 0:
			ts.Resize((0 == rows_a)? 0: 1, cols_a, slis_a, gros_a);
			break;
			case 1:
			ts.Resize(rows_a, (0 == cols_a)? 0: 1, slis_a, gros_a);
			break;
			case 2:
			ts.Resize(rows_a, cols_a, (0 == slis_a)? 0: 1, gros_a);
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, (0 == gros_a)? 0: 1);
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			#endif
		}
	}
	if (0 != numel_a) {
		TDAT* data_ts;
		TDAT* data_a;
		TDAT* data_m;
		int numel_ts;
		int num;
		switch (dim) {
			case 0:
			m.Resize(1, cols_a, slis_a, gros_a);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + 0 * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= rows_a;
			ts.Resize(1, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + 0 * cols_a + c];
							data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == rows_a)? rows_a: (rows_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = (data_ts[n] / num);
			}
			break;
			case 1:
			m.Resize(rows_a, 1, slis_a, gros_a);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + r * 1 + 0] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= cols_a;
			ts.Resize(rows_a, 1, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + r * 1 + 0];
							data_ts[g_base_ts + s_base_ts + r * 1 + 0] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == cols_a)? cols_a: (cols_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = (data_ts[n] / num);
			}
			break;
			case 2:
			m.Resize(rows_a, cols_a, 1, gros_a);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= slis_a;
			ts.Resize(rows_a, cols_a, 1, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + r * cols_a + c];
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == slis_a)? slis_a: (slis_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = (data_ts[n] / num);
			}
			break;
			case 3:
			m.Resize(rows_a, cols_a, slis_a, 1);
			m.Zeros();
			data_m = m.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_m = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_m = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_m[g_base_m + s_base_m + r * cols_a + c] += data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
			m /= gros_a;
			ts.Resize(rows_a, cols_a, slis_a, 1);
			ts.Zeros();
			data_ts = ts.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							TDAT variance = data_a[g_base_a + s_base_a + r * cols_a + c] - data_m[g_base_ts + s_base_ts + r * cols_a + c];
							data_ts[g_base_ts + s_base_ts + r * cols_a + c] += variance * variance;
						}
					}
				}
			}
			num = ("0" == ddof || 1 == gros_a)? gros_a: (gros_a - 1);
			numel_ts = ts.numel();
			for (int n = 0; n < numel_ts; ++n) {
				data_ts[n] = (data_ts[n] / num);
			}
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int n = 0; n < numel_a; ++n) {
				data_ts[n] = data_a[n];
			}
			#endif
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MaxTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	ts.Resize(1, 1, 1, 1);
	ts.Zeros();
	int numel_a = a.numel();
	if (0 != numel_a) {
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();
		TDAT maximum = data_a[0];
		for (int n = 0; n < numel_a; ++n) {
			if (data_a[n] > maximum) {
				maximum = data_a[n];
			}
		}
		data_ts[0] = maximum;
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MaxTemplate(const TensorTemplate<TDAT>& a, int dim, TensorTemplate<TDAT>* pos) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		switch (dim) {
			case 0:
			ts.Resize((0 == rows_a)? 0: 1, cols_a, slis_a, gros_a);
			break;
			case 1:
			ts.Resize(rows_a, (0 == cols_a)? 0: 1, slis_a, gros_a);
			break;
			case 2:
			ts.Resize(rows_a, cols_a, (0 == slis_a)? 0: 1, gros_a);
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, (0 == gros_a)? 0: 1);
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			#endif
		}
	}
	if (0 != numel_a) {
		TDAT temp = 0;
		TDAT* data_ts;
		TDAT* data_a;
		switch (dim) {
			case 0:
			ts.Resize(1, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == r) {
								data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp > data_ts[g_base_ts + s_base_ts + 0 * cols_a + c]) {
								data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(1, cols_a, slis_a, gros_a);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = g * 1 * cols_a * slis_a;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = s * 1 * cols_a;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + 0 * cols_a + c]) {
									data_p[g_base_ts + s_base_ts + 0 * cols_a + c] = r;
								}
							}
						}
					}
				}
			}
			break;
			case 1:
			ts.Resize(rows_a, 1, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == c) {
								data_ts[g_base_ts + s_base_ts + r * 1 + 0] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp > data_ts[g_base_ts + s_base_ts + r * 1 + 0]) {
								data_ts[g_base_ts + s_base_ts + r * 1 + 0] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(rows_a, 1, slis_a, gros_a);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = g * rows_a * 1 * slis_a;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = s * rows_a * 1;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + r * 1 + 0]) {
									data_p[g_base_ts + s_base_ts + r * 1 + 0] = c;
								}
							}
						}
					}
				}
			}
			break;
			case 2:
			ts.Resize(rows_a, cols_a, 1, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == s) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp > data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(rows_a, cols_a, 1, gros_a);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = g * rows_a * cols_a * 1;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = 0;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
									data_p[g_base_ts + s_base_ts + r * cols_a + c] = s;
								}
							}
						}
					}
				}
			}
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, 1);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == g) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp > data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(rows_a, cols_a, slis_a, 1);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = 0;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = s * rows_a * cols_a;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
									data_p[g_base_ts + s_base_ts + r * cols_a + c] = g;
								}
							}
						}
					}
				}
			}
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int n = 0; n < numel_a; ++n) {
				data_ts[n] = data_a[n];
			}
			#endif
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MinTemplate(const TensorTemplate<TDAT>& a) {
	TensorTemplate<TDAT> ts;
	ts.Resize(1, 1, 1, 1);
	ts.Zeros();
	int numel_a = a.numel();
	if (0 != numel_a) {
		TDAT* data_ts = ts.data();
		TDAT* data_a = a.data();
		TDAT minimum = data_a[0];
		for (int n = 0; n < numel_a; ++n) {
			if (data_a[n] < minimum) {
				minimum = data_a[n];
			}
		}
		data_ts[0] = minimum;
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MinTemplate(const TensorTemplate<TDAT>& a, int dim, TensorTemplate<TDAT>* pos) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		switch (dim) {
			case 0:
			ts.Resize((0 == rows_a)? 0: 1, cols_a, slis_a, gros_a);
			break;
			case 1:
			ts.Resize(rows_a, (0 == cols_a)? 0: 1, slis_a, gros_a);
			break;
			case 2:
			ts.Resize(rows_a, cols_a, (0 == slis_a)? 0: 1, gros_a);
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, (0 == gros_a)? 0: 1);
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			#endif
		}
	}
	if (0 != numel_a) {
		TDAT temp = 0;
		TDAT* data_ts;
		TDAT* data_a;
		switch (dim) {
			case 0:
			ts.Resize(1, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * 1 * cols_a * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * 1 * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == r) {
								data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp < data_ts[g_base_ts + s_base_ts + 0 * cols_a + c]) {
								data_ts[g_base_ts + s_base_ts + 0 * cols_a + c] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(1, cols_a, slis_a, gros_a);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = g * 1 * cols_a * slis_a;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = s * 1 * cols_a;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + 0 * cols_a + c]) {
									data_p[g_base_ts + s_base_ts + 0 * cols_a + c] = r;
								}
							}
						}
					}
				}
			}
			break;
			case 1:
			ts.Resize(rows_a, 1, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * 1 * slis_a;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * 1;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == c) {
								data_ts[g_base_ts + s_base_ts + r * 1 + 0] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp < data_ts[g_base_ts + s_base_ts + r * 1 + 0]) {
								data_ts[g_base_ts + s_base_ts + r * 1 + 0] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(rows_a, 1, slis_a, gros_a);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = g * rows_a * 1 * slis_a;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = s * rows_a * 1;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + r * 1 + 0]) {
									data_p[g_base_ts + s_base_ts + r * 1 + 0] = c;
								}
							}
						}
					}
				}
			}
			break;
			case 2:
			ts.Resize(rows_a, cols_a, 1, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = g * rows_a * cols_a * 1;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = 0;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == s) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp < data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(rows_a, cols_a, 1, gros_a);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = g * rows_a * cols_a * 1;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = 0;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
									data_p[g_base_ts + s_base_ts + r * cols_a + c] = s;
								}
							}
						}
					}
				}
			}
			break;
			case 3:
			ts.Resize(rows_a, cols_a, slis_a, 1);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_ts = 0;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_ts = s * rows_a * cols_a;
					int s_base_a = s * rows_a * cols_a;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							if (0 == g) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
							temp = data_a[g_base_a + s_base_a + r * cols_a + c];
							if (temp < data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
								data_ts[g_base_ts + s_base_ts + r * cols_a + c] = temp;
							}
						}
					}
				}
			}
			if (nullptr != pos) {
				pos->Resize(rows_a, cols_a, slis_a, 1);
				pos->Zeros();
				TDAT* data_p = pos->data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_ts = 0;
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_ts = s * rows_a * cols_a;
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								temp = data_a[g_base_a + s_base_a + r * cols_a + c];
								if (temp == data_ts[g_base_ts + s_base_ts + r * cols_a + c]) {
									data_p[g_base_ts + s_base_ts + r * cols_a + c] = g;
								}
							}
						}
					}
				}
			}
			break;
			default:
			#ifdef TENSOR_DEBUG
			T_ERROR("Dimension should be 0, 1, 2, or 3.\n");
			#else
			ts.Resize(rows_a, cols_a, slis_a, gros_a);
			ts.Zeros();
			data_ts = ts.data();
			data_a = a.data();
			for (int n = 0; n < numel_a; ++n) {
				data_ts[n] = data_a[n];
			}
			#endif
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MMTemplate(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& b) {
	int numel_a = a.numel();
	int numel_b = b.numel();
	if (1 == numel_a) {
		TensorTemplate<TDAT> c(b);
		TDAT* data_a = a.data();
		c *= data_a[0];
		return c;
	}
	if (1 == numel_b) {
		TensorTemplate<TDAT> c(a);
		TDAT* data_b = b.data();
		c *= data_b[0];
		return c;
	}
	else {
		int rows_a = a.rows();
		int rows_b = b.rows();
		int cols_a = a.cols();
		int cols_b = b.cols();
		int slis_a = a.slis();
		int slis_b = b.slis();
		int gros_a = a.gros();
		int gros_b = b.gros();
		TensorTemplate<TDAT> c;
		if (slis_a > 1 || slis_b > 1 || gros_a > 1 || gros_b > 1) {
			T_ERROR("Tensors should be 2-D, or one of them should be 0-D.\n");
		}
		else if (cols_a != rows_b) {
			T_ERROR("Dimension mismatched.\n");
		}
		else {
			c.Resize(rows_a, cols_b);
			c.Zeros();
			int numel_c = c.numel();
			if (0 != numel_c && 0 != numel_a && 0 != numel_b) {
				TDAT* data_a = a.data();
				TDAT* data_b = b.data();
				TDAT* data_c = c.data();
				for (int r = 0; r < rows_a; ++r) {
					for (int c = 0; c < cols_b; ++c) {
						TDAT temp = 0;
						for (int k = 0; k < cols_a; ++k) {
							temp = temp + data_a[r * cols_a + k] * data_b[k * cols_b + c];
						}
						data_c[r * cols_b + c] = temp;
					}
				}
			}
		}
		return c;
	}
}


template <typename TDAT>
inline TensorTemplate<TDAT> WhereTemplate(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& b, const TensorTemplate<TDAT>& c) {
	int numel_a = a.numel();
	int numel_b = b.numel();
	int numel_c = c.numel();
	if (MatchTemplate<TDAT>(a, b) && MatchTemplate<TDAT>(b, c)) {
		TensorTemplate<TDAT> d;
		d.Resize(a.rows(), a.cols(), a.slis(), a.gros());
		if (0 != numel_a) {
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			auto data_d = d.data();
			for (int n = 0; n < numel_a; ++n) {
				data_d[n] = (data_a[n]? data_b[n]: data_c[n]);
			}
		}
		return d;
	}
	else if (1 == numel_a) {
		auto data_a = a.data();
		if (data_a[0]) {
			return b;
		}
		else {
			return c;
		}
	}
	else if (1 == numel_b && MatchTemplate<TDAT>(a, c)) {
		TensorTemplate<TDAT> d;
		d.Resize(a.rows(), a.cols(), a.slis(), a.gros());
		if (0 != numel_a) {
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			auto data_d = d.data();
			for (int n = 0; n < numel_a; ++n) {
				data_d[n] = (data_a[n]? data_b[0]: data_c[n]);
			}
		}
		return d;
	}
	else if (1 == numel_c && MatchTemplate<TDAT>(a, b)) {
		TensorTemplate<TDAT> d;
		d.Resize(a.rows(), a.cols(), a.slis(), a.gros());
		if (0 != numel_a) {
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			auto data_d = d.data();
			for (int n = 0; n < numel_a; ++n) {
				data_d[n] = (data_a[n]? data_b[n]: data_c[0]);
			}
		}
		return d;
	}
	else if (1 == numel_b && 1 == numel_c) {
		TensorTemplate<TDAT> d;
		d.Resize(a.rows(), a.cols(), a.slis(), a.gros());
		if (0 != numel_a) {
			auto data_a = a.data();
			auto data_b = b.data();
			auto data_c = c.data();
			auto data_d = d.data();
			for (int n = 0; n < numel_a; ++n) {
				data_d[n] = (data_a[n]? data_b[0]: data_c[0]);
			}
		}
		return d;
	}
	else {
		T_ERROR("Dimension mismatched.\n");
		return c;
	}
}


template <typename TDAT>
inline TensorTemplate<TDAT> CatTemplate(const std::vector<TensorTemplate<TDAT> >& v, int dim) {
	TensorTemplate<TDAT> ts;
	if (dim > 3 || dim < 0) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Dimension for splitting should be 0, 1, 2, or 3.\n");
		#else
		T_WARNING("Dimension for splitting should be 0, 1, 2, or 3.\n");
		#endif
		return ts;
	}
	else {
		if (0 == dim) {
			int num_tensors = v.size();
			if (0 != num_tensors) {
				int rows_ts = 0;
				int cols_ts = v[0].cols();
				int slis_ts = v[0].slis();
				int gros_ts = v[0].gros();
				for (int n = 0; n < num_tensors; ++n) {
					if (cols_ts == v[n].cols() && slis_ts == v[n].slis() && gros_ts == v[n].gros()) {
						rows_ts += v[n].rows();
					}
					else {
						#ifdef TENSOR_DEBUG
						T_ERROR("Dimension mismatched.\n");
						#else
						num_tensors = n;
						#endif
					}
				}
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				TDAT* data_ts = ts.data();
				int ra = 0;
				for (int n = 0; n < num_tensors; ++n) {
					int r_vn = 0;
					int rows_vn = v[n].rows();
					TDAT* data_vn = v[n].data();
					for (int g = 0; g < gros_ts; ++g) {
						int g_base_ts = g * rows_ts * cols_ts * slis_ts;
						for (int s = 0; s < slis_ts; ++s) {
							int s_base_ts = s * rows_ts * cols_ts;
							for (int r = 0; r < rows_vn; ++r) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g_base_ts + s_base_ts + (r + ra) * cols_ts + c] = data_vn[r_vn++];
								}
							}
						}
					}
					ra += rows_vn;
				}
			}
			return ts;
		}
		else if (1 == dim) {
			int num_tensors = v.size();
			if (0 != num_tensors) {
				int rows_ts = v[0].rows();
				int cols_ts = 0;
				int slis_ts = v[0].slis();
				int gros_ts = v[0].gros();
				for (int n = 0; n < num_tensors; ++n) {
					if (rows_ts == v[n].rows() && slis_ts == v[n].slis() && gros_ts == v[n].gros()) {
						cols_ts += v[n].cols();
					}
					else {
						#ifdef TENSOR_DEBUG
						T_ERROR("Dimension mismatched.\n");
						#else
						num_tensors = n;
						#endif
					}
				}
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				TDAT* data_ts = ts.data();
				int ca = 0;
				for (int n = 0; n < num_tensors; ++n) {
					int c_vn = 0;
					int cols_vn = v[n].cols();
					TDAT* data_vn = v[n].data();
					for (int g = 0; g < gros_ts; ++g) {
						int g_base_ts = g * rows_ts * cols_ts * slis_ts;
						for (int s = 0; s < slis_ts; ++s) {
							int s_base_ts = s * rows_ts * cols_ts;
							for (int r = 0; r < rows_ts; ++r) {
								for (int c = 0; c < cols_vn; ++c) {
									data_ts[g_base_ts + s_base_ts + r * cols_ts + (c + ca)] = data_vn[c_vn++];
								}
							}
						}
					}
					ca += cols_vn;
				}
			}
			return ts;
		}
		else if (2 == dim) {
			int num_tensors = v.size();
			if (0 != num_tensors) {
				int rows_ts = v[0].rows();
				int cols_ts = v[0].cols();
				int slis_ts = 0;
				int gros_ts = v[0].gros();
				for (int n = 0; n < num_tensors; ++n) {
					if (rows_ts == v[n].rows() && cols_ts == v[n].cols() && gros_ts == v[n].gros()) {
						slis_ts += v[n].slis();
					}
					else {
						#ifdef TENSOR_DEBUG
						T_ERROR("Dimension mismatched.\n");
						#else
						num_tensors = n;
						#endif
					}
				}
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				TDAT* data_ts = ts.data();
				int sa = 0;
				for (int n = 0; n < num_tensors; ++n) {
					int s_vn = 0;
					int slis_vn = v[n].slis();
					TDAT* data_vn = v[n].data();
					for (int g = 0; g < gros_ts; ++g) {
						int g_base_ts = g * rows_ts * cols_ts * slis_ts;
						for (int s = 0; s < slis_vn; ++s) {
							int s_base_ts = (s + sa) * rows_ts * cols_ts;
							for (int r = 0; r < rows_ts; ++r) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g_base_ts + s_base_ts + r * cols_ts + c] = data_vn[s_vn++];
								}
							}
						}
					}
					sa += slis_vn;
				}
			}
			return ts;
		}
		else {
			int num_tensors = v.size();
			if (0 != num_tensors) {
				int rows_ts = v[0].rows();
				int cols_ts = v[0].cols();
				int slis_ts = v[0].slis();
				int gros_ts = 0;
				for (int n = 0; n < num_tensors; ++n) {
					if (rows_ts == v[n].rows() && cols_ts == v[n].cols() && slis_ts == v[n].slis()) {
						gros_ts += v[n].gros();
					}
					else {
						#ifdef TENSOR_DEBUG
						T_ERROR("Dimension mismatched.\n");
						#else
						num_tensors = n;
						#endif
					}
				}
				ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
				TDAT* data_ts = ts.data();
				int ga = 0;
				for (int n = 0; n < num_tensors; ++n) {
					int g_vn = 0;
					int gros_vn = v[n].gros();
					TDAT* data_vn = v[n].data();
					for (int g = 0; g < gros_vn; ++g) {
						int g_base_ts = (g + ga) * rows_ts * cols_ts * slis_ts;
						for (int s = 0; s < slis_ts; ++s) {
							int s_base_ts = s * rows_ts * cols_ts;
							for (int r = 0; r < rows_ts; ++r) {
								for (int c = 0; c < cols_ts; ++c) {
									data_ts[g_base_ts + s_base_ts + r * cols_ts + c] = data_vn[g_vn++];
								}
							}
						}
					}
					ga += gros_vn;
				}
			}
			return ts;
		}
	}
}


template <typename TDAT>
inline std::vector<TensorTemplate<TDAT> > SplitTemplate(const TensorTemplate<TDAT>& a, int dim) {
	std::vector<TensorTemplate<TDAT> > v;
	if (dim > 3 || dim < 0) {
		#ifdef TENSOR_DEBUG
		T_ERROR("Dimension for splitting should be 0, 1, 2, or 3.\n");
		#else
		T_WARNING("Dimension for splitting should be 0, 1, 2, or 3.\n");
		#endif
		return v;
	}
	else {
		int rows_a = a.rows();
		int cols_a = a.cols();
		int slis_a = a.slis();
		int gros_a = a.gros();
		int numel_a = a.numel();
		if (0 == dim) {
			v.resize(rows_a, ZerosTemplate<TDAT>(1, cols_a, slis_a, gros_a));
			if (0 != numel_a) {
				TDAT* data_a = a.data();
				for (int r = 0; r < rows_a; ++r) {
					TDAT* data_ts = v[r].data();
					int r_base_a = r * cols_a;
					for (int g = 0; g < gros_a; ++g) {
						int g_base_a = g * rows_a * cols_a * slis_a;
						for (int s = 0; s < slis_a; ++s) {
							int s_base_a = s * rows_a * cols_a;
							for (int c = 0; c < cols_a; ++c) {
								*data_ts++ = data_a[g_base_a + s_base_a + r_base_a + c];
							}
						}
					}
				}
			}
			return v;
		}
		else if (1 == dim) {
			v.resize(cols_a, ZerosTemplate<TDAT>(rows_a, 1, slis_a, gros_a));
			if (0 != numel_a) {
				TDAT* data_a = a.data();
				for (int c = 0; c < cols_a; ++c) {
					TDAT* data_ts = v[c].data();
					for (int g = 0; g < gros_a; ++g) {
						int g_base_a = g * rows_a * cols_a * slis_a;
						for (int s = 0; s < slis_a; ++s) {
							int s_base_a = s * rows_a * cols_a;
							for (int r = 0; r < rows_a; ++r) {
								*data_ts++ = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
						}
					}
				}
			}
			return v;
		}
		else if (2 == dim) {
			v.resize(slis_a, ZerosTemplate<TDAT>(rows_a, cols_a, 1, gros_a));
			if (0 != numel_a) {
				TDAT* data_a = a.data();
				for (int s = 0; s < slis_a; ++s) {
					TDAT* data_ts = v[s].data();
					int s_base_a = s * rows_a * cols_a;
					for (int g = 0; g < gros_a; ++g) {
						int g_base_a = g * rows_a * cols_a * slis_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								*data_ts++ = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
						}
					}
				}
			}
			return v;
		}
		else {
			v.resize(gros_a, ZerosTemplate<TDAT>(rows_a, cols_a, slis_a, 1));
			if (0 != numel_a) {
				TDAT* data_a = a.data();
				for (int g = 0; g < gros_a; ++g) {
					TDAT* data_ts = v[g].data();
					int g_base_a = g * rows_a * cols_a * slis_a;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_a = s * rows_a * cols_a;
						for (int r = 0; r < rows_a; ++r) {
							for (int c = 0; c < cols_a; ++c) {
								*data_ts++ = data_a[g_base_a + s_base_a + r * cols_a + c];
							}
						}
					}
				}
			}
			return v;
		}
	}
}


template <typename TDAT>
inline TensorTemplate<TDAT> PaddingAsymTemplate(const TensorTemplate<TDAT>& a, int ra, int rb, int ca, int cb, int sa, int sb, int ga, int gb, TDAT padding_value) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int rows_ts = rows_a + ra + rb;
	int cols_ts = cols_a + ca + cb;
	int slis_ts = slis_a + sa + sb;
	int gros_ts = gros_a + ga + gb;
	if (rows_ts < 0) rows_ts = 0;
	if (cols_ts < 0) cols_ts = 0;
	if (slis_ts < 0) slis_ts = 0;
	if (gros_ts < 0) gros_ts = 0;
	ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
	int numel_ts = ts.numel();
	if (0 != numel_ts) {
		TDAT* data_a = a.data();
		TDAT* data_ts = ts.data();
		int r_c_s_a = rows_a * cols_a * slis_a;
		int r_c_a = rows_a * cols_a;
		for (int g = 0; g < gros_ts; ++g) {
			int g_base_ts = g * rows_ts * cols_ts * slis_ts;
			int g_ref = g - ga;
			for (int s = 0; s < slis_ts; ++s) {
				int s_base_ts = s * rows_ts * cols_ts;
				int s_ref = s - sa;
				for (int r = 0; r < rows_ts; ++r) {
					int r_base_ts = r * cols_ts;
					int r_ref = r - ra;
					for (int c = 0; c< cols_ts; ++c) {
						int c_ref = c - ca;
						if (c_ref < 0 || c_ref >= cols_a || r_ref < 0 || r_ref >= rows_a || s_ref < 0 || s_ref >= slis_a || g_ref < 0 || g_ref >= gros_a) {
							data_ts[g_base_ts + s_base_ts + r_base_ts + c] = padding_value;
						}
						else {
							data_ts[g_base_ts + s_base_ts + r_base_ts + c] = data_a[g_ref * r_c_s_a + s_ref * r_c_a + r_ref * cols_a + c_ref];
						}
					}
				}
			}
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> AvgPool2dTemplate(const TensorTemplate<TDAT>& a, int k, TensorTemplate<TDAT>* mask) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		if (nullptr != mask) {
			mask->Resize(rows_a, cols_a, slis_a, gros_a);
		}
	}
	else {
		if (k <= 0) {
			#ifdef TENSOR_DEBUG
			T_ERROR("Kernel size should be positive.\n");
			#else
			k = 1;
			#endif
		}
		if (1 == k) {
			if (nullptr != mask) {
				mask->Resize(rows_a, cols_a, slis_a, gros_a);
				mask->Ones();
			}
			return a;
		}
		else {
			int rows_ts = rows_a / k;
			int cols_ts = cols_a / k;
			int slis_ts = slis_a;
			int gros_ts = gros_a;
			ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
			int numel_ts = ts.numel();
			if (0 != numel_ts) {
				ts.Zeros();
				if (nullptr != mask) {
					mask->Resize(rows_a, cols_a, slis_a, gros_a);
					mask->Zeros();
					int rows_m = rows_ts * k;
					int cols_m = cols_ts * k;
					TDAT* data_m = mask->data();
					TDAT value = (typeid(float).name() == typeid(TDAT).name())? (1.0 / k / k): 1;
					for (int g = 0; g < gros_a; ++g) {
						int g_base_m = g * rows_a * cols_a * slis_a;
						for (int s = 0; s < slis_a; ++s) {
							int s_base_m = s * rows_a * cols_a;
							for (int r = 0; r < rows_m; ++r) {
								for (int c = 0; c < cols_m; ++c) {
									data_m[g_base_m + s_base_m + r * cols_a + c] = value;
								}
							}
						}
					}
				}
				TDAT* data_a = a.data();
				TDAT* data_ts = ts.data();
				for (int g = 0; g < gros_a; ++g) {
					int g_base_a = g * rows_a * cols_a * slis_a;
					int g_base_ts = g * rows_ts * cols_ts * slis_ts;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_a = s * rows_a * cols_a;
						int s_base_ts = s * rows_ts * cols_ts;
						for (int r = 0; r < rows_ts; ++r) {
							for (int c = 0; c < cols_ts; ++c) {
								for (int i = 0; i < k; ++i) {
									for (int j = 0; j < k; ++j) {
										data_ts[g_base_ts + s_base_ts + r * cols_ts +c] += \
										data_a[g_base_a + s_base_a + (r * k + i) * cols_a + (c * k + j)];
									}
								}
								data_ts[g_base_ts + s_base_ts + r * cols_ts +c] /= (k * k);
							}
						}
					}
				}
			}
			else {
				if (nullptr != mask) {
					mask->Resize(rows_a, cols_a, slis_a, gros_a);
					mask->Zeros();
				}
			}
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> MaxPool2dTemplate(const TensorTemplate<TDAT>& a, int k, TensorTemplate<TDAT>* mask) {
	TensorTemplate<TDAT> ts;
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int numel_a = a.numel();
	if (0 == numel_a) {
		if (nullptr != mask) {
			mask->Resize(rows_a, cols_a, slis_a, gros_a);
		}
	}
	else {
		if (k <= 0) {
			#ifdef TENSOR_DEBUG
			T_ERROR("Kernel size should be positive.\n");
			#else
			k = 1;
			#endif
		}
		if (1 == k) {
			if (nullptr != mask) {
				mask->Resize(rows_a, cols_a, slis_a, gros_a);
				mask->Ones();
			}
			return a;
		}
		else {
			int rows_ts = rows_a / k;
			int cols_ts = cols_a / k;
			int slis_ts = slis_a;
			int gros_ts = gros_a;
			ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
			int numel_ts = ts.numel();
			if (0 != numel_ts) {
				ts.Zeros();
				TDAT* data_a = a.data();
				TDAT* data_ts = ts.data();
				TDAT* data_m = nullptr;
				if (nullptr != mask) {
					mask->Resize(rows_a, cols_a, slis_a, gros_a);
					mask->Zeros();
					data_m = mask->data();
				}
				for (int g = 0; g < gros_a; ++g) {
					int g_base_a = g * rows_a * cols_a * slis_a;
					int g_base_ts = g * rows_ts * cols_ts * slis_ts;
					for (int s = 0; s < slis_a; ++s) {
						int s_base_a = s * rows_a * cols_a;
						int s_base_ts = s * rows_ts * cols_ts;
						for (int r = 0; r < rows_ts; ++r) {
							for (int c = 0; c < cols_ts; ++c) {
								TDAT max_value = data_a[g_base_a + s_base_a + (r * k) * cols_a + (c * k)];
								for (int i = 0; i < k; ++i) {
									for (int j = 0; j < k; ++j) {
										TDAT temp = data_a[g_base_a + s_base_a + (r * k + i) * cols_a + (c * k + j)];
										if (temp > max_value) {
											max_value = temp;
										}
									}
								}
								data_ts[g_base_ts + s_base_ts + r * cols_ts +c] = max_value;
								if (nullptr != mask) {
									for (int i = 0; i < k; ++i) {
										for (int j = 0; j < k; ++j) {
											int coord = g_base_a + s_base_a + (r * k + i) * cols_a + (c * k + j);
											if (data_a[coord] == max_value) {
												data_m[coord] = 1;
											}
										}
									}
								}
							}
						}
					}
				}
			}
			else {
				if (nullptr != mask) {
					mask->Resize(rows_a, cols_a, slis_a, gros_a);
					mask->Zeros();
				}
			}
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> Conv2dBaseTemplate(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& k) {
	int rows_k = k.rows();
	int cols_k = k.cols();
	int slis_k = k.slis();
	int gros_k = k.gros();
	int numel_k = k.numel();
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	TensorTemplate<TDAT> ts;
	if (0 == numel_k) {
		T_ERROR("Convolution kernels cannot be empty.\n");
	}
	else if (slis_k != slis_a) {
		T_ERROR("Dimension mismatched.\n");
	}
	else {
		int rows_ts = rows_a + rows_k - 1;
		int cols_ts = cols_a + cols_k - 1;
		int slis_ts = gros_k;
		int gros_ts = gros_a;
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		int numel_ts = ts.numel();
		if (0 != numel_ts) {
			TDAT* data_a = a.data();
			TDAT* data_k = k.data();
			TDAT* data_ts = ts.data();
			for (int g = 0; g < gros_ts; ++g) {
				int g_base_ts = g * rows_ts * cols_ts * slis_ts;
				int g_base_a = g * rows_a * cols_a * slis_a;
				int g_base_k = g * rows_k * cols_k * slis_k;
				for (int s = 0; s < slis_ts; ++s) {
					int s_base_ts = s * rows_ts * cols_ts;
					for (int r = 0; r < rows_ts; ++r) {
						for (int c = 0; c < cols_ts; ++c) {
							TDAT temp = 0;
							for (int i = 0; i < rows_k; ++i) {
								for (int j = 0; j < cols_k; ++j) {
									int x = j + c - cols_k + 1;
									int y = i + r - rows_k + 1;
									if (x < 0 || y < 0 || x >= cols_a || y >= rows_a) {
									}
									else {
										for (int t = 0; t < slis_k; ++t) {
											temp += (data_a[g_base_a + t * cols_a * rows_a + y * cols_a + x] * data_k[g_base_k + t * cols_k * rows_k + i * cols_k + j]);
										}
									}
								}
							}
							data_ts[g_base_ts + s_base_ts + r * cols_ts + c] = temp;
						}
					}
				}
			}
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> Conv2dTemplate(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& k, int stride, int padding) {
	int rows_k = k.rows();
	int cols_k = k.cols();
	int slis_k = k.slis();
	int gros_k = k.gros();
	int numel_k = k.numel();
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int st = stride;
	TensorTemplate<TDAT> ts;
	if (0 == numel_k) {
		T_ERROR("Convolution kernels cannot be empty.\n");
	}
	else if (slis_k != slis_a) {
		T_ERROR("Dimension mismatched.\n");
	}
	else {
		if (stride < 1) {
			#ifdef TENSOR_DEBUG
			T_ERROR("Stride cannot be less than 1.\n");
			#else
			T_WARNING("Stride cannot be less than 1.\n");
			st = 1;
			#endif
		}
		int rows_ts = (rows_a + padding * 2 - rows_k) / st + 1;
		int cols_ts = (cols_a + padding * 2 - cols_k) / st + 1;
		int slis_ts = gros_k;
		int gros_ts = gros_a;
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		int numel_ts = ts.numel();
		if (0 != numel_ts) {
			TDAT* data_a = a.data();
			TDAT* data_k = k.data();
			TDAT* data_ts = ts.data();
			for (int g = 0; g < gros_ts; ++g) {
				int g_base_ts = g * rows_ts * cols_ts * slis_ts;
				int g_base_a = g * rows_a * cols_a * slis_a;
				for (int s = 0; s < slis_ts; ++s) {
					int s_base_ts = s * rows_ts * cols_ts;
					int g_base_k = s * rows_k * cols_k * slis_k;
					for (int r = 0; r < rows_ts; ++r) {
						for (int c = 0; c < cols_ts; ++c) {
							TDAT temp = 0;
							for (int i = 0; i < rows_k; ++i) {
								for (int j = 0; j < cols_k; ++j) {
									int x = j + st * c - padding;
									int y = i + st * r - padding;
									if (x < 0 || y < 0 || x >= cols_a || y >= rows_a) {
									}
									else {
										for (int t = 0; t < slis_k; ++t) {
											temp += (data_a[g_base_a + t * cols_a * rows_a + y * cols_a + x] * data_k[g_base_k + t * cols_k * rows_k + i * cols_k + j]);
										}
									}
								}
							}
							data_ts[g_base_ts + s_base_ts + r * cols_ts + c] = temp;
						}
					}
				}
			}
		}
	}
	return ts;
}


template <typename TDAT>
inline TensorTemplate<TDAT> ConvTranspose2dTemplate(const TensorTemplate<TDAT>& a, const TensorTemplate<TDAT>& k, int stride, int padding) {
	int rows_a = a.rows();
	int cols_a = a.cols();
	int slis_a = a.slis();
	int gros_a = a.gros();
	int rows_k = k.rows();
	int cols_k = k.cols();
	int slis_k = k.slis();
	int gros_k = k.gros();
	int numel_k = k.numel();
	int st = stride;
	TensorTemplate<TDAT> ts;
	if (0 == numel_k) {
		T_ERROR("Convolution kernels cannot be empty.\n");
	}
	else if (slis_k != slis_a) {
		T_ERROR("Dimension mismatched.\n");
	}
	else {
		if (stride < 1) {
			#ifdef TENSOR_DEBUG
			T_ERROR("Stride cannot be less than 1.\n");
			#else
			T_WARNING("Stride cannot be less than 1.\n");
			st = 1;
			#endif
		}
		int rows_b = st * rows_a;
		int cols_b = st * cols_a;
		int slis_b = slis_a;
		int gros_b = gros_a;
		TensorTemplate<TDAT> b;
		b.Resize(rows_b, cols_b, slis_b, gros_b);
		int numel_b = b.numel();
		if (0 != numel_b) {
			b.Zeros();
			TDAT* data_a = a.data();
			TDAT* data_b = b.data();
			for (int g = 0; g < gros_a; ++g) {
				int g_base_a = g * rows_a * cols_a * slis_a;
				int g_base_b = g * rows_b * cols_b * slis_b;
				for (int s = 0; s < slis_a; ++s) {
					int s_base_a = s * rows_a * cols_a;
					int s_base_b = s * rows_b * cols_b;
					for (int r = 0; r < rows_a; ++r) {
						for (int c = 0; c < cols_a; ++c) {
							data_b[g_base_b + s_base_b + (st * r) * cols_b + (st * c)] = data_a[g_base_a + s_base_a + r * cols_a + c];
						}
					}
				}
			}
		}

		TensorTemplate<TDAT> kernel = Rot90Template<TDAT>(k, 2, 2);

		int rows_ts = rows_b + rows_k - 1;
		int cols_ts = cols_b + cols_k - 1;
		int slis_ts = gros_k;
		int gros_ts = gros_b;
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		int numel_ts = ts.numel();
		if (0 != numel_ts) {
			TDAT* data_b = b.data();
			TDAT* data_k = kernel.data();
			TDAT* data_ts = ts.data();
			for (int g = 0; g < gros_ts; ++g) {
				int g_base_ts = g * rows_ts * cols_ts * slis_ts;
				int g_base_b = g * rows_b * cols_b * slis_b;
				for (int s = 0; s < slis_ts; ++s) {
					int s_base_ts = s * rows_ts * cols_ts;
					int g_base_k = s * rows_k * cols_k * slis_k;
					for (int r = 0; r < rows_ts; ++r) {
						for (int c = 0; c < cols_ts; ++c) {
							TDAT temp = 0;
							for (int i = 0; i < rows_k; ++i) {
								for (int j = 0; j < cols_k; ++j) {
									int x = j + c - cols_k + 1;
									int y = i + r - rows_k + 1;
									if (x < 0 || y < 0 || x >= cols_b || y >= rows_b) {
									}
									else {
										for (int t = 0; t < slis_k; ++t) {
											temp += (data_b[g_base_b + t * cols_b * rows_b + y * cols_b + x] * data_k[g_base_k + t * cols_k * rows_k + i * cols_k + j]);
										}
									}
								}
							}
							data_ts[g_base_ts + s_base_ts + r * cols_ts + c] = temp;
						}
					}
				}
			}
		}
	}
	return PaddingAsymTemplate<TDAT>(ts, -padding, -padding -st + 1, -padding, -padding - st + 1, 0, 0, 0, 0, 0);
}


template <typename TDAT>
void SaveTensorTemplate(std::string file_name, const TensorTemplate<TDAT>& ts) {
	std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
	if(!output_file) {
		#ifdef TENSOR_DEBUG
		T_ERROR("File creation failed.\n");
		#endif
	}
	else {
		int32_t cell_type = 0;
		int32_t cell_x = 1;
		int32_t cell_y = 1;
		output_file.write((char*)&cell_type, sizeof(int32_t));
		output_file.write((char*)&cell_x, sizeof(int32_t));
		output_file.write((char*)&cell_y, sizeof(int32_t));
		if (typeid(float).name() == typeid(TDAT).name()) {
			int32_t type_id = -2;
			output_file.write((char*)&type_id, sizeof(int32_t));
			int32_t rows_ts = ts.rows();
			int32_t cols_ts = ts.cols();
			int32_t slis_ts = ts.slis();
			int32_t gros_ts = ts.gros();
			int numel_ts = ts.numel();
			TDAT* data_ts = ts.data();
			output_file.write((char*)&rows_ts, sizeof(int32_t));
			output_file.write((char*)&cols_ts, sizeof(int32_t));
			output_file.write((char*)&slis_ts, sizeof(int32_t));
			output_file.write((char*)&gros_ts, sizeof(int32_t));
			float* buffer = new float[numel_ts];
			for (int i = 0; i < numel_ts; ++i) {
				buffer[i] = data_ts[i];
			}
			output_file.write((char*)buffer, numel_ts * sizeof(float));
			if (nullptr != buffer) {
				delete [] buffer;
				buffer = nullptr;
			}
		}
		else if (typeid(int32_t).name() == typeid(TDAT).name()) {
			int32_t type_id = -1;
			output_file.write((char*)&type_id, sizeof(int32_t));
			int32_t rows_ts = ts.rows();
			int32_t cols_ts = ts.cols();
			int32_t slis_ts = ts.slis();
			int32_t gros_ts = ts.gros();
			int numel_ts = ts.numel();
			TDAT* data_ts = ts.data();
			output_file.write((char*)&rows_ts, sizeof(int32_t));
			output_file.write((char*)&cols_ts, sizeof(int32_t));
			output_file.write((char*)&slis_ts, sizeof(int32_t));
			output_file.write((char*)&gros_ts, sizeof(int32_t));
			int32_t* buffer = new int32_t[numel_ts];
			for (int i = 0; i < numel_ts; ++i) {
				buffer[i] = data_ts[i];
			}
			output_file.write((char*)buffer, numel_ts * sizeof(int32_t));
			if (nullptr != buffer) {
				delete [] buffer;
				buffer = nullptr;
			}
		}
		else {
			#ifdef TENSOR_DEBUG
			T_ERROR("File creation failed.\n");
			#endif
		}
	}
	output_file.close();
}


template <typename TDAT>
void SaveTensorsTemplate(std::string file_name, const std::vector<TensorTemplate<TDAT> >& tensor_group) {
	if (tensor_group.size() == 0) {
		return;
	}
	std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
	if(!output_file) {
		#ifdef TENSOR_DEBUG
		T_ERROR("File creation failed.\n");
		#endif
	}
	else {
		int32_t num_tensors = tensor_group.size();
		int32_t cell_type = 1;
		int32_t cell_x = num_tensors;
		int32_t cell_y = 1;
		output_file.write((char*)&cell_type, sizeof(int32_t));
		output_file.write((char*)&cell_x, sizeof(int32_t));
		output_file.write((char*)&cell_y, sizeof(int32_t));

		if (typeid(float).name() == typeid(TDAT).name()) {
			int32_t type_id = -2;
			output_file.write((char*)&type_id, sizeof(int32_t));
			for (int32_t cell_iter = 0; cell_iter < num_tensors; ++ cell_iter) {
				int32_t rows_ts = tensor_group[cell_iter].rows();
				int32_t cols_ts = tensor_group[cell_iter].cols();
				int32_t slis_ts = tensor_group[cell_iter].slis();
				int32_t gros_ts = tensor_group[cell_iter].gros();
				int numel_ts = tensor_group[cell_iter].numel();
				TDAT* data_ts = tensor_group[cell_iter].data();
				output_file.write((char*)&rows_ts, sizeof(int32_t));
				output_file.write((char*)&cols_ts, sizeof(int32_t));
				output_file.write((char*)&slis_ts, sizeof(int32_t));
				output_file.write((char*)&gros_ts, sizeof(int32_t));
				float* buffer = new float[numel_ts];
				for (int i = 0; i < numel_ts; ++i) {
					buffer[i] = data_ts[i];
				}
				output_file.write((char*)buffer, numel_ts * sizeof(float));
				if (nullptr != buffer) {
					delete [] buffer;
					buffer = nullptr;
				}
			}
		}
		else if (typeid(int32_t).name() == typeid(TDAT).name()) {
			int32_t type_id = -1;
			output_file.write((char*)&type_id, sizeof(int32_t));
			for (int32_t cell_iter = 0; cell_iter < num_tensors; ++ cell_iter) {
				int32_t rows_ts = tensor_group[cell_iter].rows();
				int32_t cols_ts = tensor_group[cell_iter].cols();
				int32_t slis_ts = tensor_group[cell_iter].slis();
				int32_t gros_ts = tensor_group[cell_iter].gros();
				int numel_ts = tensor_group[cell_iter].numel();
				TDAT* data_ts = tensor_group[cell_iter].data();
				output_file.write((char*)&rows_ts, sizeof(int32_t));
				output_file.write((char*)&cols_ts, sizeof(int32_t));
				output_file.write((char*)&slis_ts, sizeof(int32_t));
				output_file.write((char*)&gros_ts, sizeof(int32_t));
				int32_t* buffer = new int32_t[numel_ts];
				for (int i = 0; i < numel_ts; ++i) {
					buffer[i] = data_ts[i];
				}
				output_file.write((char*)buffer, numel_ts * sizeof(int32_t));
				if (nullptr != buffer) {
					delete [] buffer;
					buffer = nullptr;
				}
			}
		}
		else {
			#ifdef TENSOR_DEBUG
			T_ERROR("Type NOT supported.\n");
			#endif
		}
	}
	output_file.close();
}


template <typename TDAT>
TensorTemplate<TDAT> LoadTensorTemplate(std::string file_name) {
	TensorTemplate<TDAT> ts;
	std::ifstream input_file(file_name, std::ios::in | std::ios::binary);
	if(!input_file) {
		#ifdef TENSOR_DEBUG
		T_ERROR("File does NOT exist.\n");
		#endif
	}
	else {
		int32_t cell_type = 0;
		int32_t cell_x = 0;
		int32_t cell_y = 0;
		int32_t type_id = 0;
		input_file.read((char *)&cell_type, sizeof(int32_t));
		input_file.read((char *)&cell_x, sizeof(int32_t));
		input_file.read((char *)&cell_y, sizeof(int32_t));
		input_file.read((char *)&type_id, sizeof(int32_t));

		int32_t rows_ts = 0;
		int32_t cols_ts = 0;
		int32_t slis_ts = 0;
		int32_t gros_ts = 0;
		input_file.read((char *)&rows_ts, sizeof(int32_t));
		input_file.read((char *)&cols_ts, sizeof(int32_t));
		input_file.read((char *)&slis_ts, sizeof(int32_t));
		input_file.read((char *)&gros_ts, sizeof(int32_t));
		ts.Resize(rows_ts, cols_ts, slis_ts, gros_ts);
		int numel_ts = ts.numel();
		TDAT* data_ts = ts.data();
		if (type_id == -2) {
			float* buffer = new float[numel_ts];
			input_file.read((char *)buffer, numel_ts * sizeof(float));
			for (int i = 0; i < numel_ts; ++i) {
				data_ts[i] = buffer[i];
			}
			if (nullptr != buffer) {
				delete [] buffer;
				buffer = nullptr;
			}
		}
		else if (type_id == -1) {
			int32_t* buffer = new int32_t[numel_ts];
			input_file.read((char *)buffer, numel_ts * sizeof(int32_t));
			for (int i = 0; i < numel_ts; ++i) {
				data_ts[i] = buffer[i];
			}
			if (nullptr != buffer) {
				delete [] buffer;
				buffer = nullptr;
			}
		}
		else {
			#ifdef TENSOR_DEBUG
			T_ERROR("Type error in file loading.\n");
			#endif
		}
	}
	input_file.close();
	return ts;
}


template <typename TDAT>
std::vector<TensorTemplate<TDAT> > LoadTensorsTemplate(std::string file_name) {
	std::vector<TensorTemplate<TDAT> > tensor_group;
	std::ifstream input_file(file_name, std::ios::in | std::ios::binary);
	if(!input_file) {
		#ifdef TENSOR_DEBUG
		T_ERROR("File does NOT exist.\n");
		#endif
	}
	else {
		int32_t num_tensors = 1;
		int32_t cell_type = 0;
		int32_t cell_x = 0;
		int32_t cell_y = 0;
		int32_t type_id = 0;
		input_file.read((char *)&cell_type, sizeof(int32_t));
		input_file.read((char *)&cell_x, sizeof(int32_t));
		input_file.read((char *)&cell_y, sizeof(int32_t));
		input_file.read((char *)&type_id, sizeof(int32_t));
		if (cell_type) {
			num_tensors = cell_x * cell_y;
		}
		for (int32_t cell_iter = 0; cell_iter < num_tensors; ++cell_iter) {
			int32_t rows_ts = 0;
			int32_t cols_ts = 0;
			int32_t slis_ts = 0;
			int32_t gros_ts = 0;
			input_file.read((char *)&rows_ts, sizeof(int32_t));
			input_file.read((char *)&cols_ts, sizeof(int32_t));
			input_file.read((char *)&slis_ts, sizeof(int32_t));
			input_file.read((char *)&gros_ts, sizeof(int32_t));
			TensorTemplate<TDAT> ts = RawTemplate<TDAT>(rows_ts, cols_ts, slis_ts, gros_ts);
			int numel_ts = ts.numel();
			TDAT* data_ts = ts.data();
			if (type_id == -2) {
				float* buffer = new float[numel_ts];
				input_file.read((char *)buffer, numel_ts * sizeof(float));
				for (int i = 0; i < numel_ts; ++i) {
					data_ts[i] = buffer[i];
				}
				if (nullptr != buffer) {
					delete [] buffer;
					buffer = nullptr;
				}
			}
			else if (type_id == -1) {
				int32_t* buffer = new int32_t[numel_ts];
				input_file.read((char *)buffer, numel_ts * sizeof(int32_t));
				for (int i = 0; i < numel_ts; ++i) {
					data_ts[i] = buffer[i];
				}
				if (nullptr != buffer) {
					delete [] buffer;
					buffer = nullptr;
				}
			}
			else {
				#ifdef TENSOR_DEBUG
				T_ERROR("Type error in file loading.\n");
				#endif
			}
			tensor_group.push_back(ts);
		}
	}
	input_file.close();
	return tensor_group;
}


TensorTemplate<int32_t> LoadBmp(const std::string& strFile, const std::string& option = "");
bool SaveBmp(const std::string& strFile, const TensorTemplate<int32_t>& ts, const std::string& option = "", uint16_t color_bit = 32);


// ****************************************
// Tensor of 32-bit integers
// ****************************************
namespace itensor32 {
typedef TensorTemplate<int32_t> Tensor;
Tensor Raw(int rows = 1, int cols = 1, int slis = 1, int gros = 1);
Tensor Raw(const Tensor& a);
Tensor Zeros(int rows = 1, int cols = 1, int slis = 1, int gros = 1);
Tensor Zeros(const Tensor& a);
Tensor Ones(int rows = 1, int cols = 1, int slis = 1, int gros = 1);
Tensor Ones(const Tensor& a);
Tensor Arange(int num);
bool Match(const Tensor& a, const Tensor& b);
int Numel(const Tensor& a);
TensorTemplate<int32_t> Size(const Tensor& a);
Tensor Reshape(const Tensor& a, int rows, int cols = 1, int slis = 1, int gros = 1);
Tensor Transpose(const Tensor& a);
Tensor Flip(const Tensor& a, int dim = 1);
Tensor Flip(const Tensor& a, const std::string& dim_string);
Tensor Repmat(const Tensor& a, int rt = 1, int ct = 1, int st = 1, int gt = 1);
Tensor Kron(const Tensor& a, const Tensor& b);
Tensor Permute(const Tensor& a, int dim0, int dim1, int dim2 = 2, int dim3 = 3);
Tensor Rot90(const Tensor& a, int times = 1, int axis = 2);
Tensor Rearrange(const Tensor& a, const std::vector<int>& v, int dim);
Tensor Sum(const Tensor& a);
Tensor Sum(const Tensor& a, int dim);
Tensor Mean(const Tensor& a);
Tensor Mean(const Tensor& a, int dim);
Tensor Stddev(const Tensor& a, const std::string& ddof = "0");
Tensor Stddev(const Tensor& a, int dim, const std::string& ddof = "0");
Tensor Var(const Tensor& a, const std::string& ddof = "0");
Tensor Var(const Tensor& a, int dim, const std::string& ddof = "0");
Tensor Max(const Tensor& a);
Tensor Max(const Tensor& a, int dim, Tensor* pos = nullptr);
Tensor Min(const Tensor& a);
Tensor Min(const Tensor& a, int dim, Tensor* pos = nullptr);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor MM(const Tensor& a, const Tensor& b);
Tensor Where(const Tensor& a, const Tensor& b, const Tensor& c);
Tensor Cat(const std::vector<Tensor>& v, int dim);
std::vector<Tensor> Split(const Tensor& a, int dim);
Tensor PaddingAsym(const Tensor& a, int ra, int rb, int ca = 0, int cb = 0, int sa = 0, int sb = 0, int ga = 0, int gb = 0, int32_t padding_value = 0);
Tensor Padding(const Tensor& a, int ra, int ca = 0, int sa = 0, int ga = 0, int32_t padding_value = 0);
Tensor AvgPool2d(const Tensor& a, int k = 2, Tensor* mask = nullptr);
Tensor MaxPool2d(const Tensor& a, int k = 2, Tensor* mask = nullptr);
Tensor Conv2dBase(const Tensor& a, const Tensor& k);
Tensor Conv2d(const Tensor& a, const Tensor& k, int stride = 1, int padding = 0);
Tensor ConvTranspose2d(const Tensor& a, const Tensor& k, int stride = 1, int padding = 0);
void SaveTensor(std::string file_name, const Tensor& ts);
void SaveTensors(std::string file_name, const std::vector<Tensor>& tensor_group);
Tensor LoadTensor(std::string file_name);
std::vector<Tensor> LoadTensors(std::string file_name);
Tensor Magic(int rows);
Tensor Rand(int rows = 1, int cols = 1, int slis = 1, int gros = 1, int32_t upper = 10, int32_t lower = 0);

DECLARE_FUNC_T(Abs);
DECLARE_FUNC_T(Sign);
DECLARE_FUNC_T_S(operator%, int32_t);
DECLARE_FUNC_S_T(operator%, int32_t);
DECLARE_FUNC_T_T(operator%, int32_t);
DECLARE_FUNC_T_S(Mod, int32_t);
DECLARE_FUNC_S_T(Mod, int32_t);
DECLARE_FUNC_T_T(Mod, int32_t);
DECLARE_FUNC_T_S(Pow, int32_t);
DECLARE_FUNC_S_T(Pow, int32_t);
DECLARE_FUNC_T_T(Pow, int32_t);
DECLARE_FUNC_T_S(operator+, int32_t);
DECLARE_FUNC_S_T(operator+, int32_t);
DECLARE_FUNC_T_T(operator+, int32_t);
DECLARE_FUNC_T_S(operator-, int32_t);
DECLARE_FUNC_S_T(operator-, int32_t);
DECLARE_FUNC_T_T(operator-, int32_t);
DECLARE_FUNC_T_S(operator*, int32_t);
DECLARE_FUNC_S_T(operator*, int32_t);
DECLARE_FUNC_T_S(Mul, int32_t);
DECLARE_FUNC_S_T(Mul, int32_t);
DECLARE_FUNC_T_T(Mul, int32_t);
DECLARE_FUNC_T_S(operator/, int32_t);
DECLARE_FUNC_S_T(operator/, int32_t);
DECLARE_FUNC_T_S(Div, int32_t);
DECLARE_FUNC_S_T(Div, int32_t);
DECLARE_FUNC_T_T(Div, int32_t);
DECLARE_FUNC_T_S(operator>, int32_t);
DECLARE_FUNC_S_T(operator>, int32_t);
DECLARE_FUNC_T_T(operator>, int32_t);
DECLARE_FUNC_T_S(operator<, int32_t);
DECLARE_FUNC_S_T(operator<, int32_t);
DECLARE_FUNC_T_T(operator<, int32_t);
DECLARE_FUNC_T_S(operator==, int32_t);
DECLARE_FUNC_S_T(operator==, int32_t);
DECLARE_FUNC_T_T(operator==, int32_t);
DECLARE_FUNC_T_S(operator>=, int32_t);
DECLARE_FUNC_S_T(operator>=, int32_t);
DECLARE_FUNC_T_T(operator>=, int32_t);
DECLARE_FUNC_T_S(operator<=, int32_t);
DECLARE_FUNC_S_T(operator<=, int32_t);
DECLARE_FUNC_T_T(operator<=, int32_t);
DECLARE_FUNC_T_S(operator!=, int32_t);
DECLARE_FUNC_S_T(operator!=, int32_t);
DECLARE_FUNC_T_T(operator!=, int32_t);
DECLARE_FUNC_T(Logic);
DECLARE_FUNC_T(operator++);
DECLARE_FUNC_T(operator--);
DECLARE_FUNC_T(operator-);
DECLARE_FUNC_T(IsNaN);
DECLARE_FUNC_T(IsInf);
DECLARE_FUNC_T(IsFinite);

// For itensor32 only
DECLARE_FUNC_T(operator~);
DECLARE_FUNC_T_S(operator<<, int32_t);
DECLARE_FUNC_S_T(operator<<, int32_t);
DECLARE_FUNC_T_T(operator<<, int32_t);
DECLARE_FUNC_T_S(operator>>, int32_t);
DECLARE_FUNC_S_T(operator>>, int32_t);
DECLARE_FUNC_T_T(operator>>, int32_t);
DECLARE_FUNC_T_S(operator&, int32_t);
DECLARE_FUNC_S_T(operator&, int32_t);
DECLARE_FUNC_T_T(operator&, int32_t);
DECLARE_FUNC_T_S(operator|, int32_t);
DECLARE_FUNC_S_T(operator|, int32_t);
DECLARE_FUNC_T_T(operator|, int32_t);
DECLARE_FUNC_T_S(And, int32_t);
DECLARE_FUNC_S_T(And, int32_t);
DECLARE_FUNC_T_T(And, int32_t);
DECLARE_FUNC_T_S(Or, int32_t);
DECLARE_FUNC_S_T(Or, int32_t);
DECLARE_FUNC_T_T(Or, int32_t);
DECLARE_FUNC_T_S(Nand, int32_t);
DECLARE_FUNC_S_T(Nand, int32_t);
DECLARE_FUNC_T_T(Nand, int32_t);
DECLARE_FUNC_T_S(Nor, int32_t);
DECLARE_FUNC_S_T(Nor, int32_t);
DECLARE_FUNC_T_T(Nor, int32_t);
DECLARE_FUNC_T_S(Xor, int32_t);
DECLARE_FUNC_S_T(Xor, int32_t);
DECLARE_FUNC_T_T(Xor, int32_t);
DECLARE_FUNC_T_S(Xnor, int32_t);
DECLARE_FUNC_S_T(Xnor, int32_t);
DECLARE_FUNC_T_T(Xnor, int32_t);
}  // namespace itensor32


// ****************************************
// Tensor of single precision floating point numbers
// ****************************************
namespace ftensor {
typedef TensorTemplate<float> Tensor;
Tensor Raw(int rows = 1, int cols = 1, int slis = 1, int gros = 1);
Tensor Raw(const Tensor& a);
Tensor Zeros(int rows = 1, int cols = 1, int slis = 1, int gros = 1);
Tensor Zeros(const Tensor& a);
Tensor Ones(int rows = 1, int cols = 1, int slis = 1, int gros = 1);
Tensor Ones(const Tensor& a);
Tensor Arange(int num);
bool Match(const Tensor& a, const Tensor& b);
int Numel(const Tensor& a);
TensorTemplate<int32_t> Size(const Tensor& a);
Tensor Reshape(const Tensor& a, int rows, int cols = 1, int slis = 1, int gros = 1);
Tensor Transpose(const Tensor& a);
Tensor Flip(const Tensor& a, int dim = 1);
Tensor Flip(const Tensor& a, const std::string& dim_string);
Tensor Repmat(const Tensor& a, int rt = 1, int ct = 1, int st = 1, int gt = 1);
Tensor Kron(const Tensor& a, const Tensor& b);
Tensor Permute(const Tensor& a, int dim0, int dim1, int dim2 = 2, int dim3 = 3);
Tensor Rot90(const Tensor& a, int times = 1, int axis = 2);
Tensor Rearrange(const Tensor& a, const std::vector<int>& v, int dim);
Tensor Sum(const Tensor& a);
Tensor Sum(const Tensor& a, int dim);
Tensor Mean(const Tensor& a);
Tensor Mean(const Tensor& a, int dim);
Tensor Stddev(const Tensor& a, const std::string& ddof = "0");
Tensor Stddev(const Tensor& a, int dim, const std::string& ddof = "0");
Tensor Var(const Tensor& a, const std::string& ddof = "0");
Tensor Var(const Tensor& a, int dim, const std::string& ddof = "0");
Tensor Max(const Tensor& a);
Tensor Max(const Tensor& a, int dim, Tensor* pos = nullptr);
Tensor Min(const Tensor& a);
Tensor Min(const Tensor& a, int dim, Tensor* pos = nullptr);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor MM(const Tensor& a, const Tensor& b);
Tensor Where(const Tensor& a, const Tensor& b, const Tensor& c);
Tensor Cat(const std::vector<Tensor>& v, int dim);
std::vector<Tensor> Split(const Tensor& a, int dim);
Tensor PaddingAsym(const Tensor& a, int ra, int rb, int ca = 0, int cb = 0, int sa = 0, int sb = 0, int ga = 0, int gb = 0, float padding_value = 0.0);
Tensor Padding(const Tensor& a, int ra, int ca = 0, int sa = 0, int ga = 0, float padding_value = 0.0);
Tensor AvgPool2d(const Tensor& a, int k = 2, Tensor* mask = nullptr);
Tensor MaxPool2d(const Tensor& a, int k = 2, Tensor* mask = nullptr);
Tensor Conv2dBase(const Tensor& a, const Tensor& k);
Tensor Conv2d(const Tensor& a, const Tensor& k, int stride = 1, int padding = 0);
Tensor ConvTranspose2d(const Tensor& a, const Tensor& k, int stride = 1, int padding = 0);
void SaveTensor(std::string file_name, const Tensor& ts);
void SaveTensors(std::string file_name, const std::vector<Tensor>& tensor_group);
Tensor LoadTensor(std::string file_name);
std::vector<Tensor> LoadTensors(std::string file_name);
Tensor Magic(int rows);
Tensor Rand(int rows = 1, int cols = 1, int slis = 1, int gros = 1, float upper = 1.0, float lower = 0.0);

DECLARE_FUNC_T(Abs);
DECLARE_FUNC_T(Sign);
DECLARE_FUNC_T_S(operator%, float);
DECLARE_FUNC_S_T(operator%, float);
DECLARE_FUNC_T_T(operator%, float);
DECLARE_FUNC_T_S(Mod, float);
DECLARE_FUNC_S_T(Mod, float);
DECLARE_FUNC_T_T(Mod, float);
DECLARE_FUNC_T_S(Pow, float);
DECLARE_FUNC_S_T(Pow, float);
DECLARE_FUNC_T_T(Pow, float);
DECLARE_FUNC_T_S(operator+, float);
DECLARE_FUNC_S_T(operator+, float);
DECLARE_FUNC_T_T(operator+, float);
DECLARE_FUNC_T_S(operator-, float);
DECLARE_FUNC_S_T(operator-, float);
DECLARE_FUNC_T_T(operator-, float);
DECLARE_FUNC_T_S(operator*, float);
DECLARE_FUNC_S_T(operator*, float);
DECLARE_FUNC_T_S(Mul, float);
DECLARE_FUNC_S_T(Mul, float);
DECLARE_FUNC_T_T(Mul, float);
DECLARE_FUNC_T_S(operator/, float);
DECLARE_FUNC_S_T(operator/, float);
DECLARE_FUNC_T_S(Div, float);
DECLARE_FUNC_S_T(Div, float);
DECLARE_FUNC_T_T(Div, float);
DECLARE_FUNC_T_S(operator>, float);
DECLARE_FUNC_S_T(operator>, float);
DECLARE_FUNC_T_T(operator>, float);
DECLARE_FUNC_T_S(operator<, float);
DECLARE_FUNC_S_T(operator<, float);
DECLARE_FUNC_T_T(operator<, float);
DECLARE_FUNC_T_S(operator==, float);
DECLARE_FUNC_S_T(operator==, float);
DECLARE_FUNC_T_T(operator==, float);
DECLARE_FUNC_T_S(operator>=, float);
DECLARE_FUNC_S_T(operator>=, float);
DECLARE_FUNC_T_T(operator>=, float);
DECLARE_FUNC_T_S(operator<=, float);
DECLARE_FUNC_S_T(operator<=, float);
DECLARE_FUNC_T_T(operator<=, float);
DECLARE_FUNC_T_S(operator!=, float);
DECLARE_FUNC_S_T(operator!=, float);
DECLARE_FUNC_T_T(operator!=, float);
DECLARE_FUNC_T(Logic);
DECLARE_FUNC_T(operator++);
DECLARE_FUNC_T(operator--);
DECLARE_FUNC_T(operator-);
DECLARE_FUNC_T(IsNaN);
DECLARE_FUNC_T(IsInf);
DECLARE_FUNC_T(IsFinite);

// For ftensor only
Tensor Randn(int rows = 1, int cols = 1, int slis = 1, int gros = 1, float mean = 0.0, float stddev = 1.0);

DECLARE_FUNC_T(Sin);
DECLARE_FUNC_T(Cos);
DECLARE_FUNC_T(Tan);
DECLARE_FUNC_T(Asin);
DECLARE_FUNC_T(Acos);
DECLARE_FUNC_T(Atan);
DECLARE_FUNC_T(Sinh);
DECLARE_FUNC_T(Cosh);
DECLARE_FUNC_T(Tanh);
DECLARE_FUNC_T(Sqrt);
DECLARE_FUNC_T(Ceil);
DECLARE_FUNC_T(Floor);
DECLARE_FUNC_T(Round);
DECLARE_FUNC_T(Trunc);
DECLARE_FUNC_T(Log);
DECLARE_FUNC_T(Log10);
DECLARE_FUNC_T(Exp);

}  // namespace ftensor


#endif  // __TENSORLIB_H__
