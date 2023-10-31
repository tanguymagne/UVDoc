#include "Matrix.h"
#include "memory.h"
#include <iostream>

using namespace std;

bool Matrix::IsDispInfo=false;

Matrix::Matrix(void)
{
	nRow=nCol=0;
	pData=NULL;
}

Matrix::Matrix(int nrow,int ncol,double* data)
{
	nRow=nrow;
	nCol=ncol;
	pData=new double[nRow*nCol];
	if(data==NULL)
		memset(pData,0,sizeof(double)*nRow*nCol);
	else
		memcpy(pData,data,sizeof(double)*nRow*nCol);
}

Matrix::Matrix(const Matrix& matrix)
{
	nRow=nCol=0;
	pData=NULL;
	copyData(matrix);
}

Matrix::~Matrix(void)
{
	releaseData();
}

void Matrix::releaseData()
{
	if(pData!=NULL)
		delete pData;
	pData=NULL;
	nRow=nCol=0;
}

void Matrix::copyData(const Matrix &matrix)
{
	if(!dimMatch(matrix))
		allocate(matrix);
	memcpy(pData,matrix.pData,sizeof(double)*nRow*nCol);
}

bool Matrix::dimMatch(const Matrix& matrix) const
{
	if(nCol==matrix.nCol && nRow==matrix.nRow)
		return true;
	else
		return false;
}

bool Matrix::dimcheck(const Matrix& matrix) const
{
	if(!dimMatch(matrix))
	{
		cout<<"The dimensions of the matrices don't match!"<<endl;
		return false;
	}
	return true;
}

void Matrix::reset()
{
	if(pData!=NULL)
		memset(pData,0,sizeof(double)*nRow*nCol);
}

void Matrix::allocate(int nrow,int ncol)
{
	releaseData();
	nRow=nrow;
	nCol=ncol;
	if(nRow*nCol>0)
		pData=new double[nRow*nCol];
}

void Matrix::allocate(const Matrix& matrix)
{
	allocate(matrix.nRow,matrix.nCol);
}

void Matrix::loadData(int _nrow, int _ncol, double *data)
{
	if(!matchDimension(_nrow,_ncol))
		allocate(_nrow,_ncol);
	memcpy(pData,data,sizeof(double)*nRow*nCol);
}

void Matrix::printMatrix()
{
	for(int i=0;i<nRow;i++)
	{
		for(int j=0;j<nCol;j++)
			cout<<pData[i*nCol+j]<<" ";
		cout<<endl;
	}
}

void Matrix::identity(int ndim)
{
	allocate(ndim,ndim);
	reset();
	for(int i=0;i<ndim;i++)
		pData[i*ndim+i]=1;
}

//--------------------------------------------------------------------------------------------------
// functions to check dimensionalities
//--------------------------------------------------------------------------------------------------
bool Matrix::checkDimRight(const Vector& vect) const
{
	if(nCol==vect.dim())
		return true;
	else
	{
		cout<<"The matrix and vector don't match in multiplication!"<<endl;
		return false;
	}
}

bool Matrix::checkDimRight(const Matrix &matrix) const
{
	if(nCol==matrix.nrow())
		return true;
	else
	{
		cout<<"The matrix and matrix don't match in multiplication!"<<endl;
		return false;
	}
}

bool Matrix::checkDimLeft(const Vector& vect) const
{
	if(nRow==vect.dim())
		return true;
	else
	{
		cout<<"The vector and matrix don't match in multiplication!"<<endl;
		return false;
	}
}

bool Matrix::checkDimLeft(const Matrix &matrix) const
{
	if(nRow==matrix.ncol())
		return true;
	else
	{
		cout<<"The matrix and matrix don't match in multiplication!"<<endl;
		return false;
	}
}

//--------------------------------------------------------------------------------------------------
// functions for numerical computation
//--------------------------------------------------------------------------------------------------
void Matrix::Multiply(Vector &result, const Vector &vect) const
{
	checkDimRight(vect);
	if(result.dim()!=nRow)
		result.allocate(nRow);
	for(int i=0;i<nRow;i++)
	{
		double temp=0;
		for(int j=0;j<nCol;j++)
			temp+=pData[i*nCol+j]*vect.data()[j];
		result.data()[i]=temp;
	}
}

void Matrix::Multiply(Matrix &result, const Matrix &matrix) const
{
	checkDimRight(matrix);
	if(!result.matchDimension(nRow,matrix.nCol))
		result.allocate(nRow,matrix.nCol);
	for(int i=0;i<nRow;i++)
		for(int j=0;j<matrix.nCol;j++)
		{
			double temp=0;
			for(int k=0;k<nCol;k++)
				temp+=pData[i*nCol+k]*matrix.pData[k*matrix.nCol+j];
			result.pData[i*matrix.nCol+j]=temp;
		}
}

void Matrix::transpose(Matrix &result) const
{
	if(!result.matchDimension(nCol,nRow))
		result.allocate(nCol,nRow);
	for(int i=0;i<nCol;i++)
		for(int j=0;j<nRow;j++)
			result.pData[i*nRow+j]=pData[j*nCol+i];
}

void Matrix::fromVector(const Vector &vect)
{
	if(!matchDimension(vect.dim(),1))
		allocate(vect.dim(),1);
	memcpy(pData,vect.data(),sizeof(double)*vect.dim());
}

double Matrix::norm2() const
{
	if(pData==NULL)
		return 0;
	double temp=0;
	for(int i=0;i<nCol*nRow;i++)
		temp+=pData[i]*pData[i];
	return temp;
}

//--------------------------------------------------------------------------------------------------
// operators
//--------------------------------------------------------------------------------------------------
Matrix& Matrix::operator=(const Matrix& matrix)
{
	copyData(matrix);
	return *this;
}

Matrix& Matrix::operator +=(double val)
{
	for(int i=0;i<nCol*nRow;i++)
		pData[i]+=val;
	return *this;
}

Matrix& Matrix::operator -=(double val)
{
	for(int i=0;i<nCol*nRow;i++)
		pData[i]-=val;
	return *this;
}

Matrix& Matrix::operator *=(double val)
{
	for(int i=0;i<nCol*nRow;i++)
		pData[i]*=val;
	return *this;
}

Matrix& Matrix::operator /=(double val)
{
	for(int i=0;i<nCol*nRow;i++)
		pData[i]/=val;
	return *this;
}

Matrix& Matrix::operator +=(const Matrix &matrix)
{
	dimcheck(matrix);
	for(int i=0;i<nCol*nRow;i++)
		pData[i]+=matrix.pData[i];
	return *this;
}

Matrix& Matrix::operator -=(const Matrix &matrix)
{
	dimcheck(matrix);
	for(int i=0;i<nCol*nRow;i++)
		pData[i]-=matrix.pData[i];
	return *this;
}

Matrix& Matrix::operator *=(const Matrix &matrix)
{
	dimcheck(matrix);
	for(int i=0;i<nCol*nRow;i++)
		pData[i]*=matrix.pData[i];
	return *this;
}

Matrix& Matrix::operator /=(const Matrix &matrix)
{
	dimcheck(matrix);
	for(int i=0;i<nCol*nRow;i++)
		pData[i]/=matrix.pData[i];
	return *this;
}

Vector operator*(const Matrix& matrix,const Vector& vect)
{
	Vector result;
	matrix.Multiply(result,vect);
	return result;
}

Matrix operator*(const Matrix& matrix1,const Matrix& matrix2)
{
	Matrix result;
	matrix1.Multiply(result,matrix2);
	return result;
}

//--------------------------------------------------------------------------------------------------
// function for conjugate gradient method
//--------------------------------------------------------------------------------------------------
void Matrix::ConjugateGradient(Vector &result, const Vector &b) const
{
	if(nCol!=nRow)
	{
		cout<<"Error: when solving Ax=b, A is not square!"<<endl;
		return;
	}
	checkDimRight(b);
	if(!result.matchDimension(b))
		result.allocate(b);

	Vector r(b),p,q;
	result.reset();

	int nIterations=nRow*5;
	Vector rou(nIterations);
	for(int k=0;k<nIterations;k++)
	{
		rou[k]=r.norm2();
		if(IsDispInfo)
			cout<<rou[k]<<endl;

		if(rou[k]<1E-20)
			break;
		if(k==0)
			p=r;
		else
		{
			double ratio=rou[k]/rou[k-1];
			p=r+p*ratio;
		}
		Multiply(q,p);
		double alpha=rou[k]/innerproduct(p,q);
		result+=p*alpha;
		r-=q*alpha;
	}
}

void Matrix::SolveLinearSystem(Vector &result, const Vector &b) const
{
	if(nCol==nRow)
	{
		ConjugateGradient(result,b);
		return;
	}
	if(nRow<nCol)
	{
		cout<<"Not enough observations for parameter estimation!"<<endl;
		return;
	}
	Matrix AT,ATA;
	transpose(AT);
	AT.Multiply(ATA,*this);
	Vector ATb;
	AT.Multiply(ATb,b);
	ATA.ConjugateGradient(result,ATb);
}

#ifdef _QT

bool Matrix::writeMatrix(QFile &file) const
{
	file.write((char *)&nRow,sizeof(int));
	file.write((char *)&nCol,sizeof(int));
	if(file.write((char *)pData,sizeof(double)*nRow*nCol)!=sizeof(double)*nRow*nCol)
		return false;
	return true;
}

bool Matrix::readMatrix(QFile &file)
{
	releaseData();
	file.read((char *)&nRow,sizeof(int));
	file.read((char *)&nCol,sizeof(int));
	if(nRow*nCol>0)
	{
		allocate(nRow,nCol);
		if(file.read((char *)pData,sizeof(double)*nRow*nCol)!=sizeof(double)*nRow*nCol)
			return false;
	}
	return true;
}
#endif

#ifdef _MATLAB

void Matrix::readMatrix(const mxArray* prhs)
{
	if(pData!=NULL)
		delete pData;
	int nElements = mxGetNumberOfDimensions(prhs);
	if(nElements>2)
		mexErrMsgTxt("A matrix is expected to be loaded!");
	const mwSize* dims = mxGetDimensions(prhs);
	allocate(dims[0],dims[1]);
	double* data = (double*)mxGetData(prhs);
	for(int i =0; i<nRow; i++)
		for(int j =0; j<nCol; j++)
			pData[i*nCol+j] = data[j*nRow+i];
}

void Matrix::writeMatrix(mxArray*& plhs) const
{
	mwSize dims[2];
	dims[0]=nRow;dims[1]=nCol;
	plhs=mxCreateNumericArray(2, dims,mxDOUBLE_CLASS, mxREAL);
	double* data = (double *)mxGetData(plhs);
	for(int i =0; i<nRow; i++)
		for(int j =0; j<nCol; j++)
			data[j*nRow+i] = pData[i*nCol+j];
}

#endif