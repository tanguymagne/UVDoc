#pragma once

#include "memory.h"
#include "Image.h"

typedef int T_state;					// T_state is the type for state
typedef unsigned char T_input;			// T_input is the data type of the input image
// T_message is the data type of the messages and beliefs
//typedef double T_message;  
#define INTMESSAGE
#ifdef INTMESSAGE
	typedef int T_message;
#else
	typedef double T_message;
#endif

//-----------------------------------------------------------------------------------
// class for 1D pixel buffer
//-----------------------------------------------------------------------------------
template <class T>
class PixelBuffer1D
{
private:
	short int nDim;
	T* pData;
public:
	PixelBuffer1D(void)
	{
		nDim=0;
		pData=NULL;
	}
	PixelBuffer1D(int ndims)
	{
		allocate(ndims);
	}
	void allocate(int ndims)
	{
		nDim=ndims;
	}
	~PixelBuffer1D()
	{
		nDim=0;
		pData=NULL;
	}
	inline const T operator [](int index) const
	{
		return pData[index];
	}
	inline T& operator [](int index)
	{
		return pData[index];
	}
	T*& data(){return pData;};
	const T* data() const{return pData;};
	int nElements() const{return nDim;};
};

//-----------------------------------------------------------------------------------
// class for 2D pixel buffer
//-----------------------------------------------------------------------------------
template <class T>
class PixelBuffer2D
{
private:
	short int nDimX,nDimY;
	T* pData;
public:
	PixelBuffer2D(void)
	{
		nDimX=nDimY=0;
		pData=NULL;
	}
	PixelBuffer2D(int ndimx,int ndimy)
	{
		allocate(ndimx,ndimy);
	}
	void allocate(int ndimx,int ndimy)
	{
		nDimX=ndimx;
		nDimY=ndimy;
		pData=NULL;
	}
	~PixelBuffer2D()
	{
		nDimX=nDimY=0;
		pData=NULL;
	}
	inline const T operator [](int index) const
	{
		return pData[index];
	}
	inline T& operator [](int index)
	{
		return pData[index];
	}
	T*& data(){return pData;};
	const T* data() const{return pData;};
	int nElements()const{return nDimX*nDimY;};
};

//-----------------------------------------------------------------------------------
// the class for BP flow
//-----------------------------------------------------------------------------------
class BPFlow
{
private:
	bool IsDisplay;
	bool IsDataTermTruncated;
	bool IsTRW;
	double CTRW;

	size_t Height,Width,Area,nChannels;
    size_t Height2,Width2;
    
	T_input *pIm1,*pIm2;   // the two images for matching

	int *pOffset[2]; // the predicted flow 
	int *pWinSize[2];// the dimension of the matching size
	size_t nTotalMatches;

	// the buffers for belief propagation
	PixelBuffer2D<T_message>* pDataTerm;					// the data term
	PixelBuffer1D<T_message>* pRangeTerm[2];               // the range term
	PixelBuffer1D<T_message>* pSpatialMessage[2];      // the spatial message
	PixelBuffer1D<T_message>* pDualMessage[2];            // the dual message between two layers
	PixelBuffer1D<T_message>* pBelief[2];							// the belief

	T_message *ptrDataTerm;
	T_message *ptrRangeTerm[2];
	T_message* ptrSpatialMessage[2];
	T_message *ptrDualMessage[2];
	T_message* ptrBelief[2];

	size_t nTotalSpatialElements[2];
	size_t nTotalDualElements[2];
	size_t nTotalBelifElements[2];

	int *pX; // the final states

	DImage mFlow; // the flow field

	int nNeighbors;
	double s,d,gamma;   // the parameters of regularization
	//double m_s,m_d;
	DImage Im_s,Im_d;  // per pixel parameterization
public:
	BPFlow(void);
	~BPFlow(void);
	void ReleaseBuffer();
	void setPara(double _s,double _d);
	void setPara(const DImage& im_s,const DImage& im_d){Im_s=im_s;Im_d=im_d;};
	void setDataTermTruncation(bool isTruncated){IsDataTermTruncated=isTruncated;};
	void setDisplay(bool isDisplay){IsDisplay=isDisplay;};
	void setTRW(bool isTRW){IsTRW=isTRW;};
	void setCTRW(double cTRW){CTRW=cTRW;};
	void LoadImages(int _width,int _height,int _nchannels,const T_input* pImage1,const T_input* pImage2);
	void LoadImages(int _width,int _height,int _nchannels,const T_input* pImage1,
                                        int _width2,int _height2,const T_input* pImage2);
    
	void setHomogeneousMRF(int winSize);
	
	template<class T>
	void LoadOffset(T* pOffsetX,T* pOffsetY);
	template <class T>
	void LoadWinSize(T* pWinSizeX,T* pWinSizeY);

	void ComputeDataTerm();
	void ComputeRangeTerm(double _gamma);
	void AllocateMessage();
	double MessagePassing(int nIterations,int nHierarchy,double* pEnergyList=NULL);
	void Bipartite(int count);
	void BP_S(int count);
	void TRW_S(int count);
	
	template<class T>
	bool InsideImage(T x,T y);

	template<class T1,class T2>
	size_t AllocateBuffer(T1*& pBuffer,size_t factor,T2*& ptrData,const int* pWinSize);

	template<class T1,class T2>
	size_t AllocateBuffer(T1*& pBuffer,T2*& ptrData,const int* pWinSize1,const int* pWinSize2);

	// this is the core function for message updating
	void UpdateSpatialMessage(int x,int y,int plane,int direction); //T_message* message);
	
	void UpdateDualMessage(int x,int y,int plane);

	template<class T>
	void Add2Message(T* message,const T* other,int nstates);

	template<class T>
	void Add2Message(T* message,const T* other,int nstates,double Coeff);

	void ComputeBelief();

	void FindOptimalSolution();
	void FindOptimalSolutionSequential();

	double GetEnergy();

	const int* x() const{return pX;};

	void ComputeVelocity();

	const DImage& flow() const{return mFlow;};
	DImage& flow(){return mFlow;};

	//------------------------------------------------------------------------
	// multi-grid belief propagation
	void generateCoarserLevel(BPFlow& bp);
	void propagateFinerLevel(BPFlow& bp);	
		
	template<class T>
	void ReduceImage(T* pOutputData,int width,int height,const T* pInputData);
};

template<class T>
void BPFlow::LoadOffset(T* pOffsetX,T* pOffsetY)
{
	for(int i=0;i<2;i++)
	{
		_Release1DBuffer(pOffset[i]);
		pOffset[i]=new T_state[Area];
	}
	for(size_t j=0;j<Area;j++)
	{
		pOffset[0][j]=pOffsetX[j];
		pOffset[1][j]=pOffsetY[j];
	}
}

template<class T>
void BPFlow::LoadWinSize(T* pWinSizeX,T* pWinSizeY)
{
	for(int i=0;i<2;i++)
	{
		_Release1DBuffer(pWinSize[i]);
		pWinSize[i]=new T_state[Area];
	}
	for(size_t j=0;j<Area;j++)
	{
		pWinSize[0][j]=pWinSizeX[j];
		pWinSize[1][j]=pWinSizeY[j];
	}
}
