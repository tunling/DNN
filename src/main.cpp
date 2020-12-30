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
//�ָ��ַ�������
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
//��ȡ�β�����ݣ�xΪ�������yΪ�������
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
	cout << "�ɹ���ȡ�β�������ļ���\"iris.csv\"" << endl;
	cout << "������" << x->h << "���β��ѵ������" << endl;
}
int main() {
	srand((unsigned)time(NULL));
	Matrix* x = new Matrix(100, 4);
	Matrix* y = new Matrix(100, 1);
	FileRead(x, y);	//��ȡ����������������

	Matrix* input = x->T();	//��ʼ������������4*100
	Matrix* output = y->T();	delete y;	//��ʼ�����������1*100
	double l_rate = (double)0.005;	//ѧϰ����
	int epochs = 100000;	//��������
	vector<double> costVec;	//���ÿ�ε�������ʧ����ֵ
	double last_cost; //���ǰ10000�ε�costֵ
	Matrix* w1 = new Matrix(5, 4, true);	//�����ʼ������㵽���ز�Ĳ�������5*4
	Matrix* w2 = new Matrix(1, 5, true);	//�����ʼ�����ز㵽�����Ĳ�������1*5
	cout << "\n��ʼѵ�����������" << endl;
	double begin = omp_get_wtime();
	for (int i = 0; i < epochs; i++) {
		//forward��ǰ���������� ����ʽ���㼴��
		Matrix* z1 = w1->mul(input);	//5*4 mul 4*100 = 5*100
		Matrix* _z1 = z1->mul(-1.0);	delete z1;	//5*100
		Matrix* exp_z1 = _z1->Exp();	delete _z1;	//5*100
		Matrix* _1exp_z1 = exp_z1->add(1.0); delete exp_z1;	//5*100
		Matrix* a1 = _1exp_z1->Back(); delete _1exp_z1;	//5*100
		Matrix* z2 = w2->mul(a1);	//1*5 mul 5*100 = 1*100
		//back�����򴫲������� ����ʽ���㼴��
		Matrix* dz2 = z2->sub(output);	delete z2;//1*100
		Matrix* abs_dz2 = dz2->Abs();	//1*100
		Matrix* power_abs_dz2 = abs_dz2->Power();	delete abs_dz2;	//1*100
		double cost = power_abs_dz2->sum() / 100.0;	delete power_abs_dz2;
		if ((i + 1) % 10000 == 0) {
			if ((i + 1) != 10000 && cost > last_cost) {	//����ʧ����ֵ���ӣ���ǰ�˳�ѵ��
				cout << "ģ������ǰ������ϣ�һ��������ѵ�����������٣�" << endl;
				break;
			}
			else {
				last_cost = cost;
			}
			cout << "��ǰ��������Ϊ" << i + 1 << "��cost = " << cost << endl;
		}
		costVec.push_back(cost);	//��ű��ε�������ʧ����ֵ
		Matrix* a1_T = a1->T();	//100*5	
		Matrix* dw2 = dz2->mul(a1_T);	delete a1_T;	//1*5	
		Matrix* w2_T = w2->T();	//5*1
		Matrix* da1 = w2_T->mul(dz2);	delete w2_T; delete dz2;//5*100
		Matrix* _a1 = a1->mul(-1.0);	//5*100
		Matrix* _1a1 = _a1->add(1.0);	//5*100
		Matrix* _dz1 = a1->mul_pos(_1a1);	delete _1a1; delete _a1; delete a1;//5*100
		Matrix* dz1 = da1->mul_pos(_dz1);	delete _dz1; delete da1;	//5*100
		Matrix* dw1 = dz1->mul(x);	delete dz1;//5*4
		//update�����²�������w1 w2��
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
	cout << "ѵ�����" << endl << endl;
	cout << "ѵ������ʱ��" << end - begin << "��" << endl;
	cout << "ѵ��ѧϰ���ʣ�" << l_rate << endl;
	cout << "ѵ������������" << epochs << endl;
	cout << "ѵ���߳�����" << NUM_THREAD << endl;
	cout << "������ʧ����ֵ��" << costVec.back() << endl << endl;
	/*
	for (int i = 0; i < costVec.size(); i++) {
		cout << costVec[i] << endl;
	}*/
	Matrix* test = new Matrix();	//�������Ծ��󣬱��������û����루��Ϊ���Լ���
	Matrix* z1 = w1->mul(test);
	Matrix* _z1 = z1->mul(-1.0);	delete z1;
	Matrix* exp_z1 = _z1->Exp();	delete _z1;
	Matrix* _1exp_z1 = exp_z1->add(1.0); delete exp_z1;
	Matrix* a1 = _1exp_z1->Back(); delete _1exp_z1;
	Matrix* z2 = w2->mul(a1);
	Matrix* result = z2->Rint(); delete z2;
	cout << "�β�ݵ�Ʒ��Ϊ��" << endl;
	result->show();
	cout << "TIP:\n0.ɽ�β\n1.��ɫ�β\n2.ά�������β" << endl;
	system("pause");
	delete w1;
	delete w2;
	return 0;
}