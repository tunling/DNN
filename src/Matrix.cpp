#include "Matrix.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cmath>
#include <omp.h>
using namespace std;
extern const int NUM_THREAD = 4;	//线程数
//构造函数，参数为矩阵高与宽（当randim==true时为矩阵随机赋值）
Matrix::Matrix(int _h, int _w, bool random) {
	h = _h;
	w = _w;
	m = new double*[_h];
	for (int i = 0; i < _h; i++) {
		m[i] = new double[_w];
		for (int j = 0; j < _w; j++) {
			if (random) {
				m[i][j] = rand() / double(RAND_MAX);
			}
			else {
				m[i][j] = 0;
			}
		}
	}
}
//默认构造函数，让用户输入矩阵
Matrix::Matrix() {
	cout << "请输入待预测鸢尾草性状组数：" << endl;
	cin >> w;
	h = 4;	//通过输入倒置，省略转置步骤
	m = new double*[h];
	cout << "请输入鸢尾草性状矩阵(" << w << "行，4列)：" << endl;
	cout << "如：6.4 2.8 5.6 2.2（正确品类为2）" << endl;
	for (int i = 0; i < h; i++) {
		m[i] = new double[w];
		for (int j = 0; j < w; j++) {
			cin >> m[i][j];
		}
	}
}
//析构函数，释放double **m
Matrix::~Matrix() {
	for (int i = 0; i < h; i++) {
		delete[] m[i];
	}
	delete[] m;
}
//矩阵加法
Matrix* Matrix::add(const Matrix* X) {
	if (w == X->w&&h == X->h) {
		Matrix* re = new Matrix(h, w);
		int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
		//设置并行块，设置并行核数为 NUM_THREAD
		{
#pragma omp for schedule(dynamic)
			//使用动态调度，以提升并行性能
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					re->m[i][j] = m[i][j] + X->m[i][j];
				}
			}
		}
		return re;
	}
	else {
		cout << "矩阵加法错误：矩阵阶数不匹配！" << endl;
		return NULL;
	}
}
//矩阵各元素加上x
Matrix* Matrix::add(const double x) {
	Matrix* re = new Matrix(h, w);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	//设置并行块，设置并行核数为 NUM_THREAD
	{
#pragma omp for schedule(dynamic)
		//使用动态调度，以提升并行性能
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				re->m[i][j] = m[i][j] + x;
			}
		}
	}
	return re;
}
//矩阵减法
Matrix* Matrix::sub(const Matrix* X) {
	if (w == X->w&&h == X->h) {
		Matrix* re = new Matrix(h, w);
		int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
		//设置并行块，设置并行核数为 NUM_THREAD
		{
#pragma omp for schedule(dynamic)
			//使用动态调度，以提升并行性能
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					re->m[i][j] = m[i][j] - X->m[i][j];
				}
			}
		}
		return re;
	}
	else {
		cout << "矩阵减法错误：矩阵阶数不匹配！" << endl;
		return NULL;
	}
}
//矩阵乘法（主要性能瓶颈，应对该运算并行化）
Matrix* Matrix::mul(const Matrix* X) {
	if (w == X->h) {//确保矩阵阶数匹配
		Matrix* re = new Matrix(h, X->w);
		int i, j, k;
#pragma omp parallel private(i, j, k) num_threads(NUM_THREAD)
//设置并行块，设置并行核数为 NUM_THREAD
		{
#pragma omp for schedule(dynamic)
//使用动态调度，以提升并行性能
			for (i = 0; i < h; i++) {
				for (j = 0; j < X->w; j++) {
					for (k = 0; k < w; k++) {
						re->m[i][j] = re->m[i][j] + m[i][k] * X->m[k][j];
					}
				}
			}
		}
		return re;
	}
	else {
		cout << "矩阵乘法错误：矩阵阶数不匹配！" << endl;
		return NULL;
	}
}
//数乘
Matrix* Matrix::mul(const double x) {
	Matrix* re = new Matrix(h, w);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	//设置并行块，设置并行核数为 NUM_THREAD
	{
#pragma omp for schedule(dynamic)
		//使用动态调度，以提升并行性能
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				re->m[i][j] = m[i][j] * x;
			}
		}
	}
	return re;
}
//矩阵按位置一一相乘
Matrix* Matrix::mul_pos(const Matrix* X) {
	if (w == X->w && h == X->h) {//确保矩阵阶数匹配
		Matrix* re = new Matrix(h, w);
		int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
		//设置并行块，设置并行核数为 NUM_THREAD
		{
#pragma omp for schedule(dynamic)
			//使用动态调度，以提升并行性能
			for (i = 0; i < h; i++) {
				for (j = 0; j < w; j++) {
					re->m[i][j] = m[i][j] * X->m[i][j];
				}
			}
		}
		return re;
	}
	else {
		cout << "矩阵按位相乘错误：矩阵阶数不匹配！" << endl;
		return NULL;
	}
}
//矩阵转置
Matrix* Matrix::T() {
	Matrix* re = new Matrix(w, h);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	//设置并行块，设置并行核数为 NUM_THREAD
	{
#pragma omp for schedule(dynamic)
		//使用动态调度，以提升并行性能
		for (i = 0; i < w; i++) {
			for (j = 0; j < h; j++) {
				re->m[i][j] = m[j][i];
			}
		}
	}
	return re;
}
//矩阵求倒数（每个元素求倒）
Matrix* Matrix::Back() {
	Matrix* re = new Matrix(h, w);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	//设置并行块，设置并行核数为 NUM_THREAD
	{
#pragma omp for schedule(dynamic)
		//使用动态调度，以提升并行性能
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				re->m[i][j] = (double)1.0 / m[i][j];
			}
		}
	}
	return re;
}
//矩阵每个元素x变为e^x
Matrix* Matrix::Exp() {
	Matrix* re = new Matrix(h, w);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	//设置并行块，设置并行核数为 NUM_THREAD
	{
#pragma omp for schedule(dynamic)
		//使用动态调度，以提升并行性能
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				re->m[i][j] = exp(m[i][j]);
			}
		}
	}
	return re;
}
//对矩阵每个元素求绝对值
Matrix* Matrix::Abs() {
	Matrix* re = new Matrix(h, w);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	//设置并行块，设置并行核数为 NUM_THREAD
	{
#pragma omp for schedule(dynamic)
		//使用动态调度，以提升并行性能
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				if (m[i][j] < 0) {
					re->m[i][j] = -m[i][j];
				}
			}
		}
	}
	return re;
}
//对矩阵每个元素求平方
Matrix* Matrix::Power() {
	Matrix* re = new Matrix(h, w);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	//设置并行块，设置并行核数为 NUM_THREAD
	{
#pragma omp for schedule(dynamic)
		//使用动态调度，以提升并行性能
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				re->m[i][j] = m[i][j] * m[i][j];
			}
		}
	}
	return re;
}
//对矩阵每个元素向远离0的方向取整
Matrix* Matrix::Rint() {
	Matrix* re = new Matrix(h, w);
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	{
#pragma omp for schedule(dynamic)
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				re->m[i][j] = rint(m[i][j]);
			}
		}
	}
	return re;
}
//矩阵元素和
double Matrix::sum() {
	double total = 0;
	int i, j;
#pragma omp parallel private(i, j) num_threads(NUM_THREAD)
	{
#pragma omp for schedule(dynamic) reduction(+:total)	//对加和进行归约操作
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				total += m[i][j];
			}
		}
	}
	return total;
}
//展示矩阵
void Matrix::show() {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (j < w - 1) {
				cout << m[i][j] << " ";
			}
			else {
				cout << m[i][j] << endl;
			}
		}
	}
}