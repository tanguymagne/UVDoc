#include "stdio.h"
#include "project.h"
#include "BPFlow.h"
#include "memory.h"
#include "math.h"
#include "stdlib.h"
#include "Stochastic.h"

BPFlow::BPFlow(void)
{
#ifdef _MATLAB
	IsDisplay=false;
#else
	IsDisplay=true;
#endif
   IsDataTermTruncated=false;
   IsTRW=false;
	CTRW=(double)1/2;
   //CTRW=0.55;
	Width=Height=Area=0;
	pIm1=pIm2=NULL;
	for(int i=0;i<2;i++)
	{
		pOffset[i]=NULL;
		pWinSize[i]=NULL;
	}
	pDataTerm=NULL;
	ptrDataTerm=NULL;
	for(int i=0;i<2;i++)
	{
		pRangeTerm[i]=pSpatialMessage[i]=pDualMessage[i]=pBelief[i]=NULL;
		ptrRangeTerm[i]=ptrSpatialMessage[i]=ptrDualMessage[i]=ptrBelief[i]=NULL;
	}
	pX=NULL;
	nNeighbors=4;
}

BPFlow::~BPFlow(void)
{
	ReleaseBuffer();
}

void BPFlow::ReleaseBuffer()
{
	_Release1DBuffer(pIm1);
	_Release1DBuffer(pIm2);

	for(int i=0;i<2;i++)
	{
		_Release1DBuffer(pOffset[i]); // release the buffer of the offset
		_Release1DBuffer(pWinSize[i]); // release the buffer of the size
	}

	_Release1DBuffer(pDataTerm);
	_Release1DBuffer(ptrDataTerm);
	for(int i=0;i<2;i++)
	{
		_Release1DBuffer(pRangeTerm[i]);
		_Release1DBuffer(ptrRangeTerm[i]);
		_Release1DBuffer(pSpatialMessage[i]);
		_Release1DBuffer(ptrSpatialMessage[i]);
		_Release1DBuffer(pDualMessage[i]);
		_Release1DBuffer(ptrDualMessage[i]);
		_Release1DBuffer(pBelief[i]);
		_Release1DBuffer(ptrBelief[i]);
	}
	_Release1DBuffer(pX);
}

//---------------------------------------------------------------------------------
// set the parameter of the model
// the regularization is a truncated L1 norm: __min(s|v(x,y)-v(x+1,y)|,d)
// sigma is used to penalize large displacement
//---------------------------------------------------------------------------------
void BPFlow::setPara(double _s,double _d)
{
	s=_s;
	d=_d;
	//printf("s: %f, d: %f\n",s,d);
	if(Width>0)
	{
		Im_s.allocate(Width,Height,2);
		Im_d.allocate(Width,Height,2);
		Im_s.setValue(s);
		Im_d.setValue(d);
	}
	else
		printf("The image dimension has not been specified! Call LoadImages() first\n");
}	

//----------------------------------------------------------------------
// function to load images
//----------------------------------------------------------------------
void BPFlow::LoadImages(int _width, int _height, int _nchannels, const T_input *pImage1, const T_input *pImage2)
{
	Width=_width;
	Height=_height;
	Area=Width*Height;
	nChannels=_nchannels;

	_Release1DBuffer(pIm1);
	_Release1DBuffer(pIm2);
	pIm1=new T_input[Width*Height*nChannels];
	pIm2=new T_input[Width*Height*nChannels];

	memcpy(pIm1,pImage1,sizeof(T_input)*Width*Height*nChannels);
	memcpy(pIm2,pImage2,sizeof(T_input)*Width*Height*nChannels);
    
    Width2=Width;
    Height2=Height;
}


void BPFlow::LoadImages(int _width, int _height, int _nchannels, const T_input *pImage1, 
                                                    int _width2,int _height2, const T_input *pImage2)
{
	Width=_width;
	Height=_height;
	Area=Width*Height;
	nChannels=_nchannels;
    Width2=_width2;
    Height2=_height2;

	_Release1DBuffer(pIm1);
	_Release1DBuffer(pIm2);
	pIm1=new T_input[Width*Height*nChannels];
	pIm2=new T_input[Width2*Height2*nChannels];

	memcpy(pIm1,pImage1,sizeof(T_input)*Width*Height*nChannels);
	memcpy(pIm2,pImage2,sizeof(T_input)*Width2*Height2*nChannels);
}

//------------------------------------------------------------------------------------------------
// function to set the homogeneous MRF parameters
// There is no offset, and the window size is identical for each pixel (winSize)
//------------------------------------------------------------------------------------------------
void BPFlow::setHomogeneousMRF(int winSize)
{
	for(int i=0;i<2;i++)
	{
		_Release1DBuffer(pOffset[i]); // release the buffer of the offset
		_Release1DBuffer(pWinSize[i]); // release the buffer of the size
		pOffset[i]=new T_state[Area];
		memset(pOffset[i],0,sizeof(T_state)*Area);

		pWinSize[i]=new T_state[Area];
		for(size_t j=0;j<Area;j++)
			pWinSize[i][j]=winSize;//+CStochastic::UniformSampling(3)-1;
	}
	// add some disturbance
	for(int i=0;i<2;i++)
		for(int j=0;j<Area;j++)
			pOffset[i][j]=CStochastic::UniformSampling(5)-2;
}


//------------------------------------------------------------------------------------------------
// function to verify whether a poit is inside image boundary or not
//------------------------------------------------------------------------------------------------
template <class T>
bool BPFlow::InsideImage(T x,T y)
{
	if(x>=0 && x<Width2 && y>=0 && y<Height2)
		return true;
	else
		return false;
}

//------------------------------------------------------------------------------------------------
// function to compute range term
//------------------------------------------------------------------------------------------------
void BPFlow::ComputeRangeTerm(double _gamma)
{
	gamma=_gamma;
	for(int i=0;i<2;i++)
	{
		_Release1DBuffer(pRangeTerm[i]);
		_Release1DBuffer(ptrRangeTerm[i]);
		AllocateBuffer(pRangeTerm[i],1,ptrRangeTerm[i],pWinSize[i]);
	}
	for(ptrdiff_t offset=0;offset<Area;offset++)
	{
		for(ptrdiff_t plane=0;plane<2;plane++)
		{
			int winsize=pWinSize[plane][offset];
			for(ptrdiff_t j=-winsize;j<=winsize;j++)
				pRangeTerm[plane][offset].data()[j+winsize]=gamma*fabs((double)j+pOffset[plane][offset]);
		}
	}
}

//------------------------------------------------------------------------------------------------
// function to compute data term
//------------------------------------------------------------------------------------------------
void BPFlow::ComputeDataTerm()
{
	// allocate the buffer for data term
	nTotalMatches=AllocateBuffer<PixelBuffer2D<T_message>,T_message>(pDataTerm,ptrDataTerm,pWinSize[0],pWinSize[1]);

	T_message HistMin,HistMax;
	double HistInterval;
	double* pHistogramBuffer;
	int nBins=20000;
	int total=0; // total is the total number of plausible matches, used to normalize the histogram
	pHistogramBuffer=new double[nBins];
	memset(pHistogramBuffer,0,sizeof(double)*nBins);
	HistMin= 32767;
	HistMax=0;
	//--------------------------------------------------------------------------------------------------
	// step 1. the first sweep to compute the data term for the visible matches
	//--------------------------------------------------------------------------------------------------
	for(ptrdiff_t i=0;i<Height;i++)			// index over y
		for(ptrdiff_t j=0;j<Width;j++)		// index over x
		{
			size_t index=i*Width+j;
			int XWinLength=pWinSize[0][index]*2+1;
			// loop over a local window
			for(ptrdiff_t k=-pWinSize[1][index];k<=pWinSize[1][index];k++)  // index over y
				for(ptrdiff_t l=-pWinSize[0][index];l<=pWinSize[0][index];l++)  // index over x
				{
					ptrdiff_t x=j+pOffset[0][index]+l;
					ptrdiff_t y=i+pOffset[1][index]+k;

					// if the point is outside the image boundary then continue
					if(!InsideImage(x,y))
						continue;
					ptrdiff_t index2=y*Width2+x;
					T_message foo=0;
					for(int n=0;n<nChannels;n++)
						foo+=abs(pIm1[index*nChannels+n]-pIm2[index2*nChannels+n]); // L1 norm
//#ifdef INTMESSAGE
//						foo+=abs(pIm1[index*nChannels+n]-pIm2[index2*nChannels+n]); // L1 norm
//#else
//						foo+=fabs(pIm1[index*nChannels+n]-pIm2[index2*nChannels+n]); // L1 norm
//#endif
						
					
					pDataTerm[index][(k+pWinSize[1][index])*XWinLength+l+pWinSize[0][index]]=foo;
					HistMin=__min(HistMin,foo);
					HistMax=__max(HistMax,foo);
					total++;
				}
		}
	// compute the histogram info
	HistInterval=(double)(HistMax-HistMin)/nBins;
	//HistInterval/=21;

	//--------------------------------------------------------------------------------------------------
	// step 2. get the histogram of the matching
	//--------------------------------------------------------------------------------------------------
	for(ptrdiff_t i=0;i<Height;i++)			// index over y
		for(ptrdiff_t j=0;j<Width;j++)		// index over x
		{
			size_t index=i*Width+j;
			int XWinLength=pWinSize[0][index]*2+1;
			// loop over a local window
			for(ptrdiff_t k=-pWinSize[1][index];k<=pWinSize[1][index];k++)  // index over y
				for(ptrdiff_t l=-pWinSize[0][index];l<=pWinSize[0][index];l++)  // index over x
				{
					ptrdiff_t x=j+pOffset[0][index]+l;
					ptrdiff_t y=i+pOffset[1][index]+k;

					// if the point is outside the image boundary then continue
					if(!InsideImage(x,y))
						continue;
					int foo=__min(pDataTerm[index][(k+pWinSize[1][index])*XWinLength+l+pWinSize[0][index]]/HistInterval,nBins-1);
					pHistogramBuffer[foo]++;
				}
		}
	for(size_t i=0;i<nBins;i++) // normalize the histogram
		pHistogramBuffer[i]/=total;

	T_message DefaultMatchingScore;
	double Prob=0;
	for(size_t i=0;i<nBins;i++)
	{
		Prob+=pHistogramBuffer[i];
		if(Prob>=0.5)//(double)Area/nTotalMatches) // find the matching score
		{
			DefaultMatchingScore=__max(i,1)*HistInterval+HistMin; 
			break;
		}
	}
	//DefaultMatchingScore=__min(100*DefaultMatchingScore,HistMax/10);
	if(IsDisplay)
#ifdef INTMESSAGE
		printf("Min: %d, Default: %d, Max: %d\n",HistMin,DefaultMatchingScore,HistMax);
#else
		printf("Min: %f, Default: %f, Max: %f\n",HistMin,DefaultMatchingScore,HistMax);
#endif

	//DefaultMatchingScore=0.1;
	//--------------------------------------------------------------------------------------------------
	// step 3. assigning the default matching score to the outside matches
	//--------------------------------------------------------------------------------------------------
	for(ptrdiff_t i=0;i<Height;i++)			// index over y
		for(ptrdiff_t j=0;j<Width;j++)		// index over x
		{
			size_t index=i*Width+j;
			int XWinLength=pWinSize[0][index]*2+1;
			// loop over a local window
			for(ptrdiff_t k=-pWinSize[1][index];k<=pWinSize[1][index];k++)  // index over y
				for(ptrdiff_t l=-pWinSize[0][index];l<=pWinSize[0][index];l++)  // index over x
				{
					ptrdiff_t x=j+pOffset[0][index]+l;
					ptrdiff_t y=i+pOffset[1][index]+k;

                    int _ptr=(k+pWinSize[1][index])*XWinLength+l+pWinSize[0][index];
					// if the point is outside the image boundary then continue
					if(!InsideImage(x,y))
						pDataTerm[index][_ptr]=DefaultMatchingScore;
                     else if (IsDataTermTruncated) // put truncaitons to the data term
                        pDataTerm[index][_ptr]=__min(pDataTerm[index][_ptr],DefaultMatchingScore);
				}
		}
	delete pHistogramBuffer;
}

//------------------------------------------------------------------------------------------------
//	function to allocate buffer for the messages
//------------------------------------------------------------------------------------------------
template <class T1,class T2>
size_t BPFlow::AllocateBuffer(T1*& pBuffer,size_t factor,T2*& ptrData,const int* pWinSize)
{
	pBuffer=new T1[Area*factor];
	size_t totalElements=0;
	for(ptrdiff_t i=0;i<Area;i++)
	{
		totalElements+=pWinSize[i]*2+1;
		for(ptrdiff_t j=0;j<factor;j++)
			pBuffer[i*factor+j].allocate(pWinSize[i]*2+1);
	}
	totalElements*=factor;
	ptrData=new T2[totalElements];
	memset(ptrData,0,sizeof(T2)*totalElements);

	T2* ptrDynamic=ptrData;
	size_t total=0;
	for(ptrdiff_t i=0;i<Area*factor;i++)
	{
		pBuffer[i].data()=ptrDynamic;
		ptrDynamic+=pBuffer[i].nElements();
		total+=pBuffer[i].nElements();
	}
	return total;
}

template<class T1,class T2>
size_t BPFlow::AllocateBuffer(T1*& pBuffer,T2*& ptrData,const int* pWinSize1,const int* pWinSize2)
{
	pBuffer=new T1[Area];
	size_t totalElements=0;
	for(ptrdiff_t i=0;i<Area;i++)
	{
		totalElements+=(pWinSize1[i]*2+1)*(pWinSize2[i]*2+1);
		pBuffer[i].allocate(pWinSize1[i]*2+1,pWinSize2[i]*2+1);
	}
	ptrData=new T2[totalElements];
	memset(ptrData,0,sizeof(T2)*totalElements);

	T2* ptrDynamic=ptrData;
	size_t total=0;
	for(ptrdiff_t i=0;i<Area;i++)
	{
		pBuffer[i].data()=ptrDynamic;
		ptrDynamic+=pBuffer[i].nElements();
		total+=pBuffer[i].nElements();
	}
	return total;
}

void BPFlow::AllocateMessage()
{
	// delete the buffers for the messages
	for(int i=0;i<2;i++)
	{
		_Release1DBuffer(pSpatialMessage[i]);
		_Release1DBuffer(ptrSpatialMessage[i]);
		_Release1DBuffer(pDualMessage[i]);
		_Release1DBuffer(ptrDualMessage[i]);
		_Release1DBuffer(pBelief[i]);
		_Release1DBuffer(ptrBelief[i]);
	}
	// allocate the buffers for the messages
	for(int i=0;i<2;i++)
	{
		nTotalSpatialElements[i]=AllocateBuffer(pSpatialMessage[i],nNeighbors,ptrSpatialMessage[i],pWinSize[i]);
		nTotalDualElements[i]=AllocateBuffer(pDualMessage[i],1,ptrDualMessage[i],pWinSize[i]);
		nTotalBelifElements[i]=AllocateBuffer(pBelief[i],1,ptrBelief[i],pWinSize[i]);
	}
}

//------------------------------------------------------------------------------------------------
// function for belief propagation
//------------------------------------------------------------------------------------------------
double BPFlow::MessagePassing(int nIterations,int nHierarchy,double* pEnergyList)
{
	AllocateMessage();
	if(nHierarchy>0)
	{
		BPFlow bp;
		generateCoarserLevel(bp);
		bp.MessagePassing(20,nHierarchy-1);
		bp.propagateFinerLevel(*this);
	}
	if(pX!=NULL)
		_Release1DBuffer(pX);
	pX=new int[Area*2];
	double energy;
	for(int count=0;count<nIterations;count++)
	{
		//Bipartite(count);
		BP_S(count);
		//TRW_S(count);
		
		//FindOptimalSolutionSequential();
		ComputeBelief();
		FindOptimalSolution();

		energy=GetEnergy();
		if(IsDisplay)
			printf("No. %d energy: %f...\n",count,energy);
		if(pEnergyList!=NULL)
			pEnergyList[count]=energy;
	}
	return energy;
}

//------------------------------------------------------------------------------------------------
// bipartite message update
//------------------------------------------------------------------------------------------------
void BPFlow::Bipartite(int count)
{
	// loop over vx and vy planes to update the message within each grid
	for (int k=0; k<2; k++)
		for (int i=0; i<Height; i++)
			for (int j=0; j<Width; j++)
			{
				// the bipartite update
				if (count%2==0 &&  (i+j)%2==k) // the even count
					continue;
				if (count%2==1 && (i+j)%2==1-k) // the odd count
					continue;

				//------------------------------------------------------------------------------------------------
				// update the message from (j,i,k) to the neighbors on the same plane
				//------------------------------------------------------------------------------------------------
				// the encoding of the direction
				//	0: left to right
				//	1: right to left
				//	2: top down
				//	3: bottom up
				for (int direction = 0; direction<4; direction++)
					UpdateSpatialMessage(j,i,k,direction);

				//-----------------------------------------------------------------------------------------------------
				// update the message from (j,i,k) to the dual node (j,i,1-k)
				//-----------------------------------------------------------------------------------------------------
				if(count%4<2)
					UpdateDualMessage(j,i,k);
			}
}

void BPFlow::BP_S(int count)
{
	int k=count%2;
	if (count%4<2) //forward update
		for(int i=0;i<Height;i++)
			for(int j=0;j<Width;j++)
			{
				UpdateSpatialMessage(j,i,k,0);
				UpdateSpatialMessage(j,i,k,2);
				if(count%8<4)
					UpdateDualMessage(j,i,k);
			}
	else // backward upate
		for(int i=Height-1;i>=0;i--)
			for(int j=Width-1;j>=0;j--)
			{
				UpdateSpatialMessage(j,i,k,1);
				UpdateSpatialMessage(j,i,k,3);
				if(count%8<4)
					UpdateDualMessage(j,i,k);
			}
}

void BPFlow::TRW_S(int count)
{
	int k=count%2;
	if (k==0) //forward update
		for(int i=0;i<Height;i++)
			for(int j=0;j<Width;j++)
			{
				for(int l=0;l<2;l++)
				{
					UpdateDualMessage(j,i,l);
					UpdateSpatialMessage(j,i,l,0);
					UpdateSpatialMessage(j,i,l,2);
				}
			}
	else // backward upate
		for(int i=Height-1;i>=0;i--)
			for(int j=Width-1;j>=0;j--)
			{
				for(int l=0;l<2;l++)
				{
					UpdateDualMessage(j,i,l);
					UpdateSpatialMessage(j,i,l,1);
					UpdateSpatialMessage(j,i,l,3);
				}					
			}
}


//------------------------------------------------------------------------------------------------
//  update the message from (x0,y0,plane) to the neighbors on the same plane
//    the encoding of the direction
//               2  |
//                   v
//    0 ------> <------- 1
//                   ^
//                3 |
//------------------------------------------------------------------------------------------------
void BPFlow::UpdateSpatialMessage(int x, int y, int plane, int direction)
{
	// eliminate impossible messages
	if (direction==0 && x==Width-1)
		return;
	if (direction==1 && x==0)
		return;
	if (direction==2 && y==Height-1)
		return;
	if (direction==3 && y==0)
		return;

	int offset=y*Width+x;
	int nStates=pWinSize[plane][offset]*2+1;

			

	T_message* message_org;
   	message_org=new T_message[nStates];

	int x1=x,y1=y; // get the destination
	switch(direction){
		case 0:
			x1++;
			s=Im_s.data()[offset*2+plane];
			d=Im_d.data()[offset*2+plane];
			break;
		case 1:
			x1--;
			s=Im_s.data()[(offset-1)*2+plane];
			d=Im_d.data()[(offset-1)*2+plane];
			break;
		case 2:
			y1++;
			s=Im_s.data()[offset*2+plane];
			d=Im_d.data()[offset*2+plane];
			break;
		case 3:
			y1--;
			s=Im_s.data()[(offset-Width)*2+plane];
			d=Im_d.data()[(offset-Width)*2+plane];
			break;
	}
	//s=m_s;
	//d=m_d;
	int offset1=y1*Width+x1;
	int nStates1=pWinSize[plane][offset1]*2+1; // get the number of states for the destination node
	int wsize=pWinSize[plane][offset];
	int wsize1=pWinSize[plane][offset1];

	T_message*& message=pSpatialMessage[plane][offset1*nNeighbors+direction].data();

	// initialize the message from the dual plane
	if(!IsTRW)
		memcpy(message_org,pDualMessage[plane][offset].data(),sizeof(T_message)*nStates);
	else
	{
		memset(message_org,0,sizeof(T_message)*nStates);
		Add2Message(message_org,pDualMessage[plane][offset].data(),nStates,CTRW);
	}

	// add the range term
	if(!IsTRW)
		Add2Message(message_org,pRangeTerm[plane][offset].data(),nStates);
	else
		Add2Message(message_org,pRangeTerm[plane][offset].data(),nStates,CTRW);
	
	// add spatial messages
	if(!IsTRW)
	{
		if(x>0 && direction!=1) // add left to right
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors].data(),nStates);
		if(x<Width-1 && direction!=0) // add right to left 
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates);
		if(y>0 && direction!=3) // add top down
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+2].data(),nStates);
		if(y<Height-1 && direction!=2) // add bottom up
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+3].data(),nStates);
	}
	else
	{
		if(x>0) // add left to right
			if(direction==1)
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors].data(),nStates,CTRW-1);
			else
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors].data(),nStates,CTRW);
		if(x<Width-1) // add right to left 
			if(direction==0)
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates,CTRW-1);
			else
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates,CTRW);
		if(y>0) // add top down
			if(direction==3)
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+2].data(),nStates,CTRW-1);
			else
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+2].data(),nStates,CTRW);
		if(y<Height-1) // add bottom up
			if(direction==2)
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+3].data(),nStates,CTRW-1);
			else
				Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+3].data(),nStates,CTRW);
	}
	// use distance transform function to impose smoothness compatibility
	T_message Min=CStochastic::Min(nStates,message_org)+d;
	for(ptrdiff_t l=1;l<nStates;l++)
		message_org[l]=__min(message_org[l],message_org[l-1]+s);
	for(ptrdiff_t l=nStates-2;l>=0;l--)
		message_org[l]=__min(message_org[l],message_org[l+1]+s);


	// transform the compatibility 
	int shift=pOffset[plane][offset1]-pOffset[plane][offset];
	if(abs(shift)>wsize+wsize1) // the shift is too big that there is no overlap
	{
		if(offset>0)
			for(ptrdiff_t l=0;l<nStates1;l++)
				message[l]=l*s;
		else
			for(ptrdiff_t l=0;l<nStates1;l++)
				message[l]=-l*s;
	}
	else
	{
		int start=__max(-wsize,shift-wsize1);
		int end=__min(wsize,shift+wsize1);
		for(ptrdiff_t i=start;i<=end;i++)
			message[i-shift+wsize1]=message_org[i+wsize];
		if(start-shift+wsize1>0)
			for(ptrdiff_t i=start-shift+wsize1-1;i>=0;i--)
				message[i]=message[i+1]+s;
		if(end-shift+wsize1<nStates1)
			for(ptrdiff_t i=end-shift+wsize1+1;i<nStates1;i++)
				message[i]=message[i-1]+s;
	}

	// put back the threshold
	for(ptrdiff_t l=0;l<nStates1;l++)
		message[l]=__min(message[l],Min);

	// normalize the message by subtracting the minimum value
	Min=CStochastic::Min(nStates1,message);
	for(ptrdiff_t l=0;l<nStates1;l++)
		message[l]-=Min;

	delete message_org;
}

template<class T>
void BPFlow::Add2Message(T* message,const T* other,int nstates)
{
	for(size_t i=0;i<nstates;i++)
		message[i]+=other[i];
}

template<class T>
void BPFlow::Add2Message(T* message,const T* other,int nstates,double Coeff)
{
	for(size_t i=0;i<nstates;i++)
		message[i]+=other[i]*Coeff;
}

//------------------------------------------------------------------------------------------------
// update dual message passing from one plane to the other
//------------------------------------------------------------------------------------------------
void BPFlow::UpdateDualMessage(int x, int y, int plane)
{
	int offset=y*Width+x;
	int offset1=offset;
	int wsize=pWinSize[plane][offset];
	int nStates=wsize*2+1;
	int wsize1=pWinSize[1-plane][offset];
	int nStates1=wsize1*2+1;

	s=Im_s.data()[offset*2+plane];
	d=Im_d.data()[offset*2+plane];
	//s=m_s;
	//d=m_d;

	T_message* message_org;
	message_org=new T_message[nStates];
	memset(message_org,0,sizeof(T_message)*nStates);
	
	// add the range term
	if(!IsTRW)
		Add2Message(message_org,pRangeTerm[plane][offset].data(),nStates);
	else
		Add2Message(message_org,pRangeTerm[plane][offset].data(),nStates,CTRW);

	// add spatial messages
	if(x>0)  //add left to right
	{
		if(!IsTRW)
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors].data(),nStates);
		else
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors].data(),nStates,CTRW);
	}
	if(x<Width-1) // add right to left
	{
		if(!IsTRW)
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates);
		else
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates,CTRW);
	}
	if(y>0) // add top down
	{
		if(!IsTRW)
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+2].data(),nStates);
		else
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+2].data(),nStates,CTRW);
	}
	if(y<Height-1) // add bottom up
	{
		if(!IsTRW)
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+3].data(),nStates);
		else
			Add2Message(message_org,pSpatialMessage[plane][offset*nNeighbors+2].data(),nStates,CTRW);
	}

	if(IsTRW)
		Add2Message(message_org,pDualMessage[plane][offset1].data(),nStates,CTRW-1);

	T_message*& message=pDualMessage[1-plane][offset1].data();

	T_message Min;
	// use the data term
	if(plane==0) // from vx plane to vy plane
		for(size_t l=0;l<nStates1;l++)
			message[l]=CStochastic::Min(nStates,pDataTerm[offset].data()+l*nStates,message_org);
	else					// from vy plane to vx plane
		for(size_t l=0;l<nStates1;l++)
		{
			Min=message_org[0]+pDataTerm[offset].data()[l];
			for(size_t h=0;h<nStates;h++)
				Min=__min(Min,message_org[h]+pDataTerm[offset].data()[h*nStates1+l]);
			message[l]=Min;
		}

	// normalize the message
	Min=CStochastic::Min(nStates1,message);
	for(size_t l=0;l<nStates;l++)
		message[l]-=Min;

	delete message_org;
}

//------------------------------------------------------------------------------------------------
// compute belief
//------------------------------------------------------------------------------------------------
void BPFlow::ComputeBelief()
{
	for(size_t plane=0;plane<2;plane++)
	{
		memset(ptrBelief[plane],0,sizeof(T_message)*nTotalBelifElements[plane]);
		for(size_t i=0;i<Height;i++)
			for(size_t j=0;j<Width;j++)
			{
				size_t offset=i*Width+j;
				T_message* belief=pBelief[plane][offset].data();
				int nStates=pWinSize[plane][offset]*2+1;
				// add range term
				Add2Message(belief,pRangeTerm[plane][offset].data(),nStates);
				// add message from the dual layer
				Add2Message(belief,pDualMessage[plane][offset].data(),nStates);
				if(j>0)
					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors].data(),nStates);
				if(j<Width-1)
					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates);
				if(i>0)
					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors+2].data(),nStates);
				if(i<Height-1)
					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors+3].data(),nStates);
			}
	}
}

void BPFlow::FindOptimalSolution()
{
	for(size_t plane=0;plane<2;plane++)
		for(size_t i=0;i<Area;i++)
		{
			int nStates=pWinSize[plane][i]*2+1;
			double Min;
			int index=0;
			T_message* belief=pBelief[plane][i].data();
			Min=belief[0];
			for(int l=1;l<nStates;l++)
				if(Min>belief[l])
				{
					Min=belief[l];
					index=l;
				}
			pX[i*2+plane]=index;
		}
}

//void BPFlow::FindOptimalSolutionSequential()
//{
//	for(size_t plane=0;plane<2;plane++)
//		memset(ptrBelief[plane],0,sizeof(T_message)*nTotalBelifElements[plane]);
//
//	for(size_t i=0;i<Height;i++)
//		for(size_t j=0;j<Width;j++)
//			for(size_t plane=0;plane<2;plane++)
//			{
//				int nStates=pWinSize[plane][i]*2+1;
//				size_t offset=i*Width+j;
//				T_message* belief=pBelief[plane][offset].data();
//				
//				// add range term
//				Add2Message(belief,pRangeTerm[plane][offset].data(),nStates);
//				// add message from the dual layer
//				Add2Message(belief,pDualMessage[plane][offset].data(),nStates);
//
//				if(j>0) // horizontal energy
//					for(int l=0;l<nStates;l++)
//						belief[l]+=__min((double)abs(l-pWinSize[plane][offset]+pOffset[plane][offset]-pX[(offset-1)*2+plane]+pWinSize[plane][offset-1]-pOffset[plane][offset+1])*s,d);
//				if(i>0) // vertical energy
//					for(int l=0;l<nStates;l++)
//						belief[l]+=__min((double)abs(l-pWinSize[plane][offset]+pOffset[plane][offset]-pX[(offset-Width)*2+plane]+pWinSize[plane][offset-Width]-pOffset[plane][offset-Width])*s,d);
//				if(j<Width-1)
//					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates);
//				if(i<Height-1)
//					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors+3].data(),nStates);
//
//				// find the minimum
//			int index=0;
//			double Min=belief[0];
//			for(int l=1;l<nStates;l++)
//				if(Min>belief[l])
//				{
//					Min=belief[l];
//					index=l;
//				}
//			pX[offset*2+plane]=index;
//			}
//}

void BPFlow::FindOptimalSolutionSequential()
{
	for(size_t plane=0;plane<2;plane++)
		memset(ptrBelief[plane],0,sizeof(T_message)*nTotalBelifElements[plane]);

	for(size_t i=0;i<Height;i++)
		for(size_t j=0;j<Width;j++)
			for(size_t k=0;k<2;k++)
			{
				size_t plane;
				if(j%2==0)
					plane=k;
				else
					plane=1-k;

				size_t offset=i*Width+j;
				int nStates=pWinSize[plane][offset]*2+1;
				T_message* belief=pBelief[plane][offset].data();
				
				// add range term
				Add2Message(belief,pRangeTerm[plane][offset].data(),nStates);

				if (k==0)
					// add message from the dual layer
					Add2Message(belief,pDualMessage[plane][offset].data(),nStates);
				else
					for(int l=0;l<nStates;l++)
					{
						if(plane==0) // if the current is horizontal plane
							belief[l]+=pDataTerm[offset].data()[pX[offset*2+1]*nStates+l];
						else   // if the current is vertical plane
						{
							int nStates1=pWinSize[1-plane][offset]*2+1;
							belief[l]+=pDataTerm[offset].data()[l*nStates1+pX[offset*2]];
						}
					}

				if(j>0) // horizontal energy
					for(int l=0;l<nStates;l++)
						belief[l]+=__min((double)abs(l-pWinSize[plane][offset]+pOffset[plane][offset]-pX[(offset-1)*2+plane]+pWinSize[plane][offset-1]-pOffset[plane][offset+1])*s,d);
				if(i>0) // vertical energy
					for(int l=0;l<nStates;l++)
						belief[l]+=__min((double)abs(l-pWinSize[plane][offset]+pOffset[plane][offset]-pX[(offset-Width)*2+plane]+pWinSize[plane][offset-Width]-pOffset[plane][offset-Width])*s,d);
				if(j<Width-1)
					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors+1].data(),nStates);
				if(i<Height-1)
					Add2Message(belief,pSpatialMessage[plane][offset*nNeighbors+3].data(),nStates);

				// find the minimum
				int index=0;
				double Min=belief[0];
				for(int l=1;l<nStates;l++)
					if(Min>belief[l])
					{
						Min=belief[l];
						index=l;
					}
				pX[offset*2+plane]=index;
			}
}

//------------------------------------------------------------------------------------------------
// function to get energy
//------------------------------------------------------------------------------------------------
double BPFlow::GetEnergy()
{
	double energy=0;
	for(size_t i=0;i<Height;i++)
		for(size_t j=0;j<Width;j++)
		{
			size_t offset=i*Width+j;
			for(size_t k=0;k<2;k++)
			{
				if(j<Width-1)
				{
					s=Im_s.data()[offset*2+k];
					d=Im_d.data()[offset*2+k];
					//s=m_s;
					//d=m_d;
					energy+=__min((double)abs(pX[offset*2+k]-pWinSize[k][offset]+pOffset[k][offset]-pX[(offset+1)*2+k]+pWinSize[k][offset+1]-pOffset[k][offset+1])*s,d);
				}
				if(i<Height-1)
				{
					s=Im_s.data()[offset*2+k];
					d=Im_d.data()[offset*2+k];
					//s=m_s;
					//d=m_d;
					energy+=__min((double)abs(pX[offset*2+k]-pWinSize[k][offset]+pOffset[k][offset]-pX[(offset+Width)*2+k]+pWinSize[k][offset+Width]-pOffset[k][offset+Width])*s,d);
				}
			}
			int vx=pX[offset*2];
			int vy=pX[offset*2+1];
			int nStates=pWinSize[0][offset]*2+1;
			energy+=pDataTerm[offset].data()[vy*nStates+vx];
			for(size_t k=0;k<2;k++)
				energy+=pRangeTerm[k][offset].data()[pX[offset*2+k]];
		}
	return energy;
}

void BPFlow::ComputeVelocity()
{
	mFlow.allocate(Width,Height,2);
	for(int i=0;i<Area;i++)
	{
		mFlow.data()[i*2]=pX[i*2]+pOffset[0][i]-pWinSize[0][i];
		mFlow.data()[i*2+1]=pX[i*2+1]+pOffset[1][i]-pWinSize[1][i];
	}
}

//------------------------------------------------------------------------------------------------
// multi-grid belie propagation
//------------------------------------------------------------------------------------------------
void BPFlow::generateCoarserLevel(BPFlow &bp)
{
	//------------------------------------------------------------------------------------------------
	// set the dimensions and parameters
	//------------------------------------------------------------------------------------------------
	bp.Width=Width/2;
	if(Width%2==1)
		bp.Width++;

	bp.Height=Height/2;
	if(Height%2==1)
		bp.Height++;

	bp.Area=bp.Width*bp.Height;
	bp.s=s;
	bp.d=d;

	DImage foo;
	Im_s.smoothing(foo);
	foo.imresize(bp.Im_s,bp.Width,bp.Height);
	Im_d.smoothing(foo);
	foo.imresize(bp.Im_d,bp.Width,bp.Height);

	bp.IsDisplay=IsDisplay;
	bp.nNeighbors=nNeighbors;

	//------------------------------------------------------------------------------------------------
	// allocate buffers
	//------------------------------------------------------------------------------------------------
	for(int i=0;i<2;i++)
	{
		bp.pOffset[i]=new int[bp.Area];
		bp.pWinSize[i]=new int[bp.Area];
		ReduceImage(bp.pOffset[i],Width,Height,pOffset[i]);
		ReduceImage(bp.pWinSize[i],Width,Height,pWinSize[i]);
	}
	//------------------------------------------------------------------------------------------------
	// generate data term
	//------------------------------------------------------------------------------------------------
	bp.nTotalMatches=bp.AllocateBuffer(bp.pDataTerm,bp.ptrDataTerm,bp.pWinSize[0],bp.pWinSize[1]);
	for(int i=0;i<bp.Height;i++)
		for(int j=0;j<bp.Width;j++)
		{
			int offset=i*bp.Width+j;
			for(int ii=0;ii<2;ii++)
				for(int jj=0;jj<2;jj++)
				{
					int y=i*2+ii;
					int x=j*2+jj;
					if(y<Height && x<Width)
					{
						int nStates=(bp.pWinSize[0][offset]*2+1)*(bp.pWinSize[1][offset]*2+1);
						for(int k=0;k<nStates;k++)
							bp.pDataTerm[offset].data()[k]+=pDataTerm[y*Width+x].data()[k];
					}
				}
		}
	//------------------------------------------------------------------------------------------------
	// generate range term
	//------------------------------------------------------------------------------------------------
	bp.ComputeRangeTerm(gamma/2);
}

void BPFlow::propagateFinerLevel(BPFlow &bp)
{
	for(int i=0;i<bp.Height;i++)
		for(int j=0;j<bp.Width;j++)
		{
			int y=i/2;
			int x=j/2;
			int nStates1=pWinSize[0][y*Width+x]*2+1;
			int nStates2=pWinSize[1][y*Width+x]*2+1;
			for(int k=0;k<2;k++)
			{
				memcpy(bp.pDualMessage[k][i*bp.Width+j].data(),pDualMessage[k][y*Width+x].data(),sizeof(T_message)*(pWinSize[k][y*Width+x]*2+1));
				for(int l=0;l<nNeighbors;l++)
					memcpy(bp.pSpatialMessage[k][(i*bp.Width+j)*nNeighbors+l].data(),pSpatialMessage[k][(y*Width+x)*nNeighbors+l].data(),sizeof(T_message)*(pWinSize[k][y*Width+x]*2+1));
			}
		}
}

template<class T>
void BPFlow::ReduceImage(T* pDstData,int width,int height,const T *pSrcData)
{
	int DstWidth=width/2;
	if(width%2==1)
		DstWidth++;
	int DstHeight=height/2;
	if(height%2==1)
		DstHeight++;
	memset(pDstData,0,sizeof(T)*DstWidth*DstHeight);
	int sum=0;
	for(int i=0;i<DstHeight;i++)
		for(int j=0;j<DstWidth;j++)
		{
			int offset=i*DstWidth+j;
			sum=0;
			for(int ii=0;ii<2;ii++)
				for(int jj=0;jj<2;jj++)
				{
					int x=j*2+jj;
					int y=i*2+ii;
					if(y<height && x<width)
					{
						pDstData[offset]+=pSrcData[y*width+x];
						sum++;
					}
				}
			pDstData[offset]/=sum;
		}
}