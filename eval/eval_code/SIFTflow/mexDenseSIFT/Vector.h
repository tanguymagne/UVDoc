#ifndef _Vector_h
#define _Vector_h

#include "stdio.h"
#include "project.h"
#ifdef _QT
	#include <QFile>
#endif

class Vector
{
private:
	int nDim;
	double* pData;
public:
	Vector(void);
	Vector(int ndim,double *data=NULL);
	Vector(const Vector& vect);
	~Vector(void);
	void releaseData();
	void allocate(int ndim);
	void allocate(const Vector& vect){allocate(vect.nDim);};
	void copyData(const Vector& vect);
	void dimcheck(const Vector& vect) const;
	void reset();
	double norm2() const;
	double sum() const;
	void printVector();

	// access the members
	const double* data() const{return (const double*)pData;};
	double* data() {return pData;};
	int dim() const {return nDim;};
	inline bool matchDimension(int _ndim) const {if(nDim==_ndim) return true;else return false;};
	inline bool matchDimension(const Vector& vect) const {return matchDimension(vect.nDim);};

	// operators
	inline double operator[](int index) const {return pData[index];};
	inline double& operator[](int index){return *(pData+index);};
	Vector& operator=(const Vector& vect);
	
	Vector& operator+=(const Vector& vect);
	Vector& operator*=(const Vector& vect);
	Vector& operator-=(const Vector& vect);
	Vector& operator/=(const Vector& vect);

	Vector& operator+=(double val);
	Vector& operator*=(double val);
	Vector& operator-=(double val);
	Vector& operator/=(double val);
	
	friend const Vector operator+(const Vector& vect1,const Vector& vect2);
	friend const Vector operator*(const Vector& vect1,const Vector& vect2);
	friend const Vector operator-(const Vector& vect1,const Vector& vect2);
	friend const Vector operator/(const Vector& vect1,const Vector& vect2);

	friend const Vector operator+(const Vector& vect1,double val);
	friend const Vector operator*(const Vector& vect1,double val);
	friend const Vector operator-(const Vector& vect1,double val);
	friend const Vector operator/(const Vector& vect1,double val);

	friend double innerproduct(const Vector& vect1,const Vector& vect2);
#ifdef _QT
	bool writeVector(QFile& file) const;
	bool readVector(QFile& file);
#endif
#ifdef _MATLAB
	void readVector(const mxArray* prhs);
	void writeVector(mxArray*& prhs) const;
#endif
};

#endif