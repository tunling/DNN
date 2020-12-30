//������
class Matrix {
public:
	Matrix(int _h, int _w, bool random = false);	//���캯��������Ϊ����������randim==trueʱΪ���������ֵ��
	Matrix();	//Ĭ�Ϲ��캯�������û��������
	~Matrix();	//�����������ͷ�double **m
	Matrix* add(const Matrix* X);	//����ӷ�
	Matrix* add(const double x);	//�����Ԫ�ؼ���x
	Matrix* sub(const Matrix* X);	//�������
	Matrix* mul(const Matrix* X);	//����˷�
	Matrix* mul(const double x);	//��������
	Matrix* mul_pos(const Matrix* x);	//����λ��һһ���
	Matrix* T();	//����ת��
	Matrix* Back();	//����������ÿ��Ԫ���󵹣�
	Matrix* Exp();	//����ÿ��Ԫ��x��Ϊe^x
	Matrix* Abs();	//�Ծ���ÿ��Ԫ�������ֵ
	Matrix* Power();	//�Ծ���ÿ��Ԫ����ƽ��
	Matrix* Rint();	//�Ծ���ÿ��Ԫ����Զ��0�ķ���ȡ��
	double sum();	//����Ԫ�غ�
	void show();	//չʾ����
	int h, w;	//��������
	double **m;	//��������
};