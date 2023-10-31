#ifndef _Matrix_h
#define _Matrix_h

#include "stdio.h"
#include "Vector.h"
#include "project.h"
#ifdef _QT
	#include <QFile>
#endif

class Matrix
{
private:
	int nRow,nCol;
	double* pData;
	static bool IsDispInfo;
public:
	Matrix(void);
	Matrix(int _nrow,int _ncol,double* data=NULL);
	Matrix(const Matrix& matrix);
	~Matrix(void);
	void releaseData();
	void copyData(const Matrix& matrix);
	void allocate(const Matrix& matrix);
	void allocate(int _nrow,int _ncol);
	void reset();
	bool dimMatch(const Matrix& matrix) const;
	bool dimcheck(const Matrix& matrix) const;
	void loadData(int _nrow,int _ncol,double* data);
	static void enableDispInfo(bool dispInfo=false){IsDispInfo=dispInfo;};
	// display the matrix
	void printMatrix();
	void identity(int ndim);

	// function to access the member variables
	inline int nrow() const{return nRow;};
	inline int ncol() const{return nCol;};
	inline double* data() {return pData;};
	inline const double* data() const {return (const double*)pData;};
	inline double operator [](int index) const{return pData[index];};
	inline double& operator[](int index) {return pData[index];};
	inline double data(int row,int col)const {return pData[row*nCol+col];};
	inline double& data(int row,int col) {return pData[row*nCol+col];};
	bool matchDimension(int _nrow,int _ncol) const {if(nRow==_nrow && nCol==_ncol) return true; else return false;};
	bool matchDimension(const Matrix& matrix) const {return matchDimension(matrix.nrow(),matrix.ncol());};

	// functions to check dimensions
	bool checkDimRight(const Vector& vector) const;
	bool checkDimRight(const Matrix& matrix) const;
	bool checkDimLeft(const Vector& vector) const;
	bool checkDimLeft(const Matrix& matrix) const;

	// functions for matrix computation
	void Multiply(Vector& result,const Vector& vect) const;
	void Multiply(Matrix& result,const Matrix& matrix) const;

	void transpose(Matrix& result) const;
	void fromVector(const Vector& vect);
	double norm2() const;

	// operators
	Matrix& operator=(const Matrix& matrix);
	
	Matrix& operator+=(double val);
	Matrix& operator-=(double val);
	Matrix& operator*=(double val);
	Matrix& operator/=(double val);
	
	Matrix& operator+=(const Matrix& matrix);
	Matrix& operator-=(const Matrix& matrix);
	Matrix& operator*=(const Matrix& matrix);
	Matrix& operator/=(const Matrix& matrix);

	friend Vector operator*(const Matrix& matrix,const Vector& vect);
	friend Matrix operator*(const Matrix& matrix1,const Matrix& matrix2);

	
	// solve linear systems
	void SolveLinearSystem(Vector& result,const Vector& b) const;
	void ConjugateGradient(Vector& result,const Vector& b) const;

#ifdef _QT
	bool writeMatrix(QFile& file) const;
	bool readMatrix(QFile& file);
#endif
#ifdef _MATLAB
	void readMatrix(const mxArray* prhs);
	void writeMatrix(mxArray*& prhs) const;
#endif
};

#endif