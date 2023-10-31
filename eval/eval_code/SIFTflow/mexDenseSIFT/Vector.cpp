#include "Vector.h"
#include "memory.h"
#include <iostream>

using namespace std;

Vector::Vector(void)
{
	nDim=0;
	pData=NULL;
}

Vector::Vector(int ndim, double *data)
{
	nDim=ndim;
	pData=new double[nDim];
	if(data!=NULL)
		memcpy(pData,data,sizeof(double)*nDim);
	else
		memset(pData,0,sizeof(double)*nDim);
}

Vector::Vector(const Vector& vect)
{
	nDim=0;
	pData=NULL;
	copyData(vect);
}

Vector::~Vector(void)
{
	releaseData();
}

void Vector::releaseData()
{
	if(pData!=NULL)
		delete pData;
	pData=NULL;
	nDim=0;
}

void Vector::allocate(int ndim)
{
	releaseData();
	nDim=ndim;
	if(nDim>0)
		pData=new double[nDim];
}

void Vector::copyData(const Vector &vect)
{
	if(nDim!=vect.nDim)
	{
		releaseData();
		nDim=vect.nDim;
		pData=new double[nDim];
	}
	memcpy(pData,vect.pData,sizeof(double)*nDim);
}

void Vector::dimcheck(const Vector &vect) const
{
	if(nDim!=vect.nDim)
		cout<<"The dimensions of the vectors don't match!"<<endl;
}

void Vector::reset()
{
	if(pData!=NULL)
		memset(pData,0,sizeof(double)*nDim);
}

double Vector::sum() const
{
	double total = 0;
	for(int i=0;i<nDim;i++)
		total += pData[i];
	return total;
}

double Vector::norm2() const
{
	double temp=0;
	for(int i=0;i<nDim;i++)
		temp+=pData[i]*pData[i];
	return temp;
}

void Vector::printVector()
{
	for(int i=0;i<nDim;i++)
		printf("%.6f ",pData[i]);
	printf("\n");
}

//----------------------------------------------------------------------------------
// operators
//----------------------------------------------------------------------------------
Vector& Vector::operator =(const Vector &vect)
{
	copyData(vect);
	return *this;
}

Vector& Vector::operator +=(const Vector &vect)
{
	dimcheck(vect);
	for(int i=0;i<nDim;i++)
		pData[i]+=vect.data()[i];
	return *this;
}

Vector& Vector::operator *=(const Vector &vect)
{
	dimcheck(vect);
	for(int i=0;i<nDim;i++)
		pData[i]*=vect.data()[i];
	return *this;
}

Vector& Vector::operator -=(const Vector &vect)
{
	dimcheck(vect);
	for(int i=0;i<nDim;i++)
		pData[i]-=vect.data()[i];
	return *this;
}

Vector& Vector::operator /=(const Vector &vect)
{
	dimcheck(vect);
	for(int i=0;i<nDim;i++)
		pData[i]/=vect.data()[i];
	return *this;
}

Vector& Vector::operator +=(double val)
{
	for(int i=0;i<nDim;i++)
		pData[i]+=val;
	return *this;
}

Vector& Vector::operator *=(double val)
{
	for(int i=0;i<nDim;i++)
		pData[i]*=val;
	return *this;
}

Vector& Vector::operator -=(double val)
{
	for(int i=0;i<nDim;i++)
		pData[i]-=val;
	return *this;
}

Vector& Vector::operator /=(double val)
{
	for(int i=0;i<nDim;i++)
		pData[i]/=val;
	return *this;
}

const Vector operator+(const Vector& vect1,const Vector& vect2)
{
	vect1.dimcheck(vect2);
	Vector result(vect1);
	result+=vect2;
	return result;
}

const Vector operator-(const Vector& vect1,const Vector& vect2)
{
	vect1.dimcheck(vect2);
	Vector result(vect1);
	result-=vect2;
	return result;
}

const Vector operator*(const Vector& vect1,const Vector& vect2)
{
	vect1.dimcheck(vect2);
	Vector result(vect1);
	result*=vect2;
	return result;
}

const Vector operator/(const Vector& vect1,const Vector& vect2)
{
	vect1.dimcheck(vect2);
	Vector result(vect1);
	result/=vect2;
	return result;
}

const Vector operator+(const Vector& vect,double val)
{
	Vector result(vect);
	result+=val;
	return result;
}

const Vector operator-(const Vector& vect,double val)
{
	Vector result(vect);
	result-=val;
	return result;
}

const Vector operator*(const Vector& vect,double val)
{
	Vector result(vect);
	result*=val;
	return result;
}

const Vector operator/(const Vector& vect,double val)
{
	Vector result(vect);
	result/=val;
	return result;
}

double innerproduct(const Vector& vect1,const Vector& vect2)
{
	vect1.dimcheck(vect2);
	double result=0;
	for(int i=0;i<vect1.nDim;i++)
		result+=vect1.pData[i]*vect2.pData[i];
	return result;
}

#ifdef _QT

bool Vector::writeVector(QFile& file) const
{
	file.write((char *)&nDim,sizeof(int));
	if(file.write((char *)pData,sizeof(double)*nDim)!=sizeof(double)*nDim)
		return false;
	return true;
}

bool Vector::readVector(QFile &file)
{
	releaseData();
	file.read((char *)&nDim,sizeof(int));
	if(nDim<0)
		return false;
	if(nDim>0)
	{
		allocate(nDim);
		if(file.read((char *)pData,sizeof(double)*nDim)!=sizeof(double)*nDim)
			return false;
	}
	return true;
}

#endif

#ifdef _MATLAB

void Vector::readVector(const mxArray* prhs)
{
	if(pData!=NULL)
		delete pData;
	int nElements = mxGetNumberOfDimensions(prhs);
	if(nElements>2)
		mexErrMsgTxt("A vector is expected to be loaded!");
	const mwSize* dims = mxGetDimensions(prhs);
	nDim = dims[0]*dims[1];
	pData = new double[nDim];
	memcpy(pData,(double*)mxGetData(prhs),sizeof(double)*nDim);
}

void Vector::writeVector(mxArray*& plhs) const
{
	mwSize dims[2];
	dims[0]=nDim;dims[1]=1;
	plhs=mxCreateNumericArray(2, dims,mxDOUBLE_CLASS, mxREAL);
    memcpy((double *)mxGetData(plhs),pData,sizeof(double)*nDim);
}

#endif