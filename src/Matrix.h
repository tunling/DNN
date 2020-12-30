//矩阵类
class Matrix {
public:
	Matrix(int _h, int _w, bool random = false);	//构造函数，参数为矩阵高与宽（当randim==true时为矩阵随机赋值）
	Matrix();	//默认构造函数，让用户输入矩阵
	~Matrix();	//析构函数，释放double **m
	Matrix* add(const Matrix* X);	//矩阵加法
	Matrix* add(const double x);	//矩阵各元素加上x
	Matrix* sub(const Matrix* X);	//矩阵减法
	Matrix* mul(const Matrix* X);	//矩阵乘法
	Matrix* mul(const double x);	//矩阵数乘
	Matrix* mul_pos(const Matrix* x);	//矩阵按位置一一相乘
	Matrix* T();	//矩阵转置
	Matrix* Back();	//矩阵求倒数（每个元素求倒）
	Matrix* Exp();	//矩阵每个元素x变为e^x
	Matrix* Abs();	//对矩阵每个元素求绝对值
	Matrix* Power();	//对矩阵每个元素求平方
	Matrix* Rint();	//对矩阵每个元素向远离0的方向取整
	double sum();	//矩阵元素和
	void show();	//展示矩阵
	int h, w;	//矩阵高与宽
	double **m;	//矩阵数组
};