#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <omp.h>
#include <Windows.h>
using namespace std;
extern const int NUM_THREAD;
//分割字符串函数
vector<double> split(const string& str, const string& delim) {
	vector<double> res;
	if (str == "") return res;
	char * strs = new char[str.length() + 1]; 
	strcpy(strs, str.c_str());
	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());
	char *p = strtok(strs, d);
	while (p) {
		string s = p;
		res.push_back((double)atof(s.c_str()));
		p = strtok(NULL, d);
	}
	delete[] strs;
	delete[] d;
	delete[] p;
	return res;
}
//读取鸢尾草数据，x为输入矩阵，y为输出矩阵
void FileRead(Matrix* x, Matrix* y) {
	ifstream csvInput("iris.csv");
	for (int i = 0; i < x->h; i++)
	{
		string line;
		getline(csvInput, line);
		vector<double> data = split(line, ",");
		for (int j = 0; j < (x->w) + (y->w); j++) {
			if (j < x->w) {
				x->m[i][j] = data[j];
			}
			else {
				y->m[i][0] = data[j];
			}
		}	
	}
	csvInput.close();
	cout << "成功读取鸢尾草数据文件：\"iris.csv\"" << endl;
	cout << "共包含" << x->h << "组鸢尾草训练数据" << endl;
}
int main() {
	srand((unsigned)time(NULL));
	Matrix* x = new Matrix(100, 4);
	Matrix* y = new Matrix(100, 1);
	FileRead(x, y);	//读取输入矩阵与输出矩阵

	Matrix* input = x->T();	//初始化神经网络输入4*100
	Matrix* output = y->T();	delete y;	//初始化神经网络输出1*100
	double l_rate = (double)0.005;	//学习速率
	int epochs = 100000;	//迭代次数
	vector<double> costVec;	//存放每次迭代的损失函数值
	double last_cost; //存放前10000次的cost值
	Matrix* w1 = new Matrix(5, 4, true);	//随机初始化输入层到隐藏层的参数矩阵5*4
	Matrix* w2 = new Matrix(1, 5, true);	//随机初始化隐藏层到输出层的参数矩阵1*5
	cout << "\n开始训练深度神经网络" << endl;
	double begin = omp_get_wtime();
	for (int i = 0; i < epochs; i++) {
		//forward（前向反馈）—— 按公式计算即可
		Matrix* z1 = w1->mul(input);	//5*4 mul 4*100 = 5*100
		Matrix* _z1 = z1->mul(-1.0);	delete z1;	//5*100
		Matrix* exp_z1 = _z1->Exp();	delete _z1;	//5*100
		Matrix* _1exp_z1 = exp_z1->add(1.0); delete exp_z1;	//5*100
		Matrix* a1 = _1exp_z1->Back(); delete _1exp_z1;	//5*100
		Matrix* z2 = w2->mul(a1);	//1*5 mul 5*100 = 1*100
		//back（反向传播）—— 按公式计算即可
		Matrix* dz2 = z2->sub(output);	delete z2;//1*100
		Matrix* abs_dz2 = dz2->Abs();	//1*100
		Matrix* power_abs_dz2 = abs_dz2->Power();	delete abs_dz2;	//1*100
		double cost = power_abs_dz2->sum() / 100.0;	delete power_abs_dz2;
		if ((i + 1) % 10000 == 0) {
			if ((i + 1) != 10000 && cost > last_cost) {	//若损失函数值增加，提前退出训练
				cout << "模型已提前收敛完毕（一般是由于训练数据量较少）" << endl;
				break;
			}
			else {
				last_cost = cost;
			}
			cout << "当前迭代次数为" << i + 1 << "：cost = " << cost << endl;
		}
		costVec.push_back(cost);	//存放本次迭代的损失函数值
		Matrix* a1_T = a1->T();	//100*5	
		Matrix* dw2 = dz2->mul(a1_T);	delete a1_T;	//1*5	
		Matrix* w2_T = w2->T();	//5*1
		Matrix* da1 = w2_T->mul(dz2);	delete w2_T; delete dz2;//5*100
		Matrix* _a1 = a1->mul(-1.0);	//5*100
		Matrix* _1a1 = _a1->add(1.0);	//5*100
		Matrix* _dz1 = a1->mul_pos(_1a1);	delete _1a1; delete _a1; delete a1;//5*100
		Matrix* dz1 = da1->mul_pos(_dz1);	delete _dz1; delete da1;	//5*100
		Matrix* dw1 = dz1->mul(x);	delete dz1;//5*4
		//update（更新参数矩阵w1 w2）
		Matrix* _dw1 = dw1->mul(l_rate);	delete dw1;//5*4
		Matrix* w11 = w1->sub(_dw1);	delete _dw1;	delete w1;	//5*4
		w1 = w11;
		Matrix* _dw2 = dw2->mul(l_rate);	delete dw2;//1*5
		Matrix* w22 = w2->sub(_dw2);	delete _dw2;	delete w2;	//1*5
		w2 = w22;
	}
	double end = omp_get_wtime();
	delete input;
	delete output;
	delete x;
	cout << "训练完成" << endl << endl;
	cout << "训练总用时：" << end - begin << "秒" << endl;
	cout << "训练学习速率：" << l_rate << endl;
	cout << "训练迭代次数：" << epochs << endl;
	cout << "训练线程数：" << NUM_THREAD << endl;
	cout << "最终损失函数值：" << costVec.back() << endl << endl;
	/*
	for (int i = 0; i < costVec.size(); i++) {
		cout << costVec[i] << endl;
	}*/
	Matrix* test = new Matrix();	//创建测试矩阵，本程序由用户输入（作为测试集）
	Matrix* z1 = w1->mul(test);
	Matrix* _z1 = z1->mul(-1.0);	delete z1;
	Matrix* exp_z1 = _z1->Exp();	delete _z1;
	Matrix* _1exp_z1 = exp_z1->add(1.0); delete exp_z1;
	Matrix* a1 = _1exp_z1->Back(); delete _1exp_z1;
	Matrix* z2 = w2->mul(a1);
	Matrix* result = z2->Rint(); delete z2;
	cout << "鸢尾草的品类为：" << endl;
	result->show();
	cout << "TIP:\n0.山鸢尾\n1.变色鸢尾\n2.维吉尼亚鸢尾" << endl;
	system("pause");
	delete w1;
	delete w2;
	return 0;
}