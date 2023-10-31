#pragma once
#include "Image.h"
#include "math.h"
#include "memory.h"
#include <vector>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif


class ImageFeature
{
public:
	ImageFeature(void);
	~ImageFeature(void);
	
	template <class T1,class T2>
	static void imSIFT(const Image<T1> imsrc, Image<T2>& imsift, int cellSize = 2, int stepSize = 1,bool IsBoundaryIncluded = false,int nBins = 8);

	template <class T1,class T2>
	static void imSIFT(const Image<T1> imsrc, Image<T2>& imsift, const vector<int> cellSizeVect, int stepSize = 1,bool IsBoundaryIncluded = false, int nBins = 8);
};

template <class T1,class T2>
void ImageFeature::imSIFT(const Image<T1> imsrc, Image<T2> &imsift,int cellSize,int stepSize,bool IsBoundaryIncluded,int nBins)
{
	if(cellSize<=0)
	{
		cout<<"The cell size must be positive!"<<endl;
		return;
	}

	// this parameter controls decay of the gradient energy falls into a bin
	// run SIFT_weightFunc.m to see why alpha = 9 is the best value
	int alpha = 9; 

	int width=imsrc.width(),height=imsrc.height(),nchannels =imsrc.nchannels();
	int nPixels = width*height;
	DImage imdx,imdy;
	// compute the derivatives;
	imsrc.dx(imdx,true);
	imsrc.dy(imdy,true);

	// get the maximum gradient over the channels and estimate the normalized gradient
	DImage magsrc(width,height,nchannels),mag(width,height),gradient(width,height,2);
	double Max;
	for(int i=0;i<nPixels;i++)
	{
		int offset = i*nchannels;
		for(int j = 0;j<nchannels;j++)
			magsrc.pData[offset+j] = sqrt(imdx.pData[offset+j]*imdx.pData[offset+j]+imdy.pData[offset+j]*imdy.pData[offset+j]);
		Max = magsrc.pData[offset];
		if(Max!=0)
		{
			gradient.pData[i*2] = imdx.pData[offset]/Max;
			gradient.pData[i*2+1] = imdy.pData[offset]/Max;
		}		
		for(int j = 1;j<nchannels;j++)
		{
			if(magsrc.pData[offset+j]>Max)
			{
				Max = magsrc.pData[offset+j];
				gradient.pData[i*2] = imdx.pData[offset+j]/Max;
				gradient.pData[i*2+1] = imdy.pData[offset+j]/Max;
			}
		}
		mag.pData[i] = Max;
	}

	// get the pixel-wise energy for each orientation band
	DImage imband(width,height,nBins);
	double theta = M_PI*2/nBins;
	double _cos,_sin,temp;
	for(int k = 0;k<nBins;k++)
	{
		_sin    = sin(theta*k);
		_cos   = cos(theta*k);
		for(int i = 0;i<nPixels; i++)
		{
			temp = __max(gradient.pData[i*2]*_cos + gradient.pData[i*2+1]*_sin,0);
			if(alpha>1)
				temp = pow(temp,alpha);
			imband.pData[i*nBins+k] = temp*mag.pData[i]; 
		}
	}

	// filter out the SIFT feature
	DImage filter(cellSize*2+1,1);
	filter[0] = filter[cellSize+1] = 0.25;
	for(int i = 1;i<cellSize+1;i++)
		filter[i+1] = 1;
	for(int i = cellSize+2;i<cellSize*2+1;i++)
		filter[i] = 0;

    DImage imband_cell;
	imband.imfilter_hv(imband_cell,filter.data(),cellSize,filter.data(),cellSize);

	// allocate buffer for the sift image
	int siftdim = nBins*16;
	int sift_width,sift_height,x_shift=0,y_shift=0;

	sift_width = width/stepSize;
	sift_height = height/stepSize;

	if(IsBoundaryIncluded == false)
	{
		sift_width = (width-4*cellSize)/stepSize;
		sift_height= (height-4*cellSize)/stepSize;
		x_shift = 2*cellSize;
		y_shift = 2*cellSize;
	}

	if(!imsift.matchDimension(sift_width,sift_height,siftdim))
		imsift.allocate(sift_width,sift_height,siftdim);

	// now sample to get SIFT image
	DImage sift_cell(siftdim,1);
	for(int i=0;i<sift_height;i++)
		for(int j =0;j<sift_width;j++)
		{
			int count = 0;
			for(int ii = -1;ii<=2;ii++)
				for(int jj=-1;jj<=2;jj++)
				{
					int y = __min(__max(y_shift+i*stepSize+ii*cellSize,0),height-1);
					int x = __min(__max(x_shift+j*stepSize+jj*cellSize,0),width-1);
					
					// the following code is the same as the above two for debugging purposes
					//int y = y_shift+i*stepSize+ii*cellSize;
					//int x = x_shift+j*stepSize+jj*cellSize;
					//if (x<0 || x>=width)
					//	x = __min(__max(x,0),width-1);
					//if (y<0 || y>=height)
					//	y= __min(__max(y,0),height-1);

					memcpy(sift_cell.pData+count*nBins,imband_cell.pData+(y*width+x)*nBins,sizeof(double)*nBins);
					count++;
				}
			// normalize the SIFT descriptor
			double mag = sqrt(sift_cell.norm2());
			int offset = (i*sift_width+j)*siftdim;
			//memcpy(imsift.pData+offset,sift_cell.pData,sizeof(double)*siftdim);
			for(int k = 0;k<siftdim;k++)
				imsift.pData[offset+k] = (unsigned char)__min(sift_cell.pData[k]/(mag+0.01)*255,255);//(unsigned char) __min(sift_cell.pData[k]/mag*512,255);
		}//*/
}


//--------------------------------------------------------------------------------------------------
// multi-scale SIFT features
//--------------------------------------------------------------------------------------------------
template <class T1,class T2>
void ImageFeature::imSIFT(const Image<T1> imsrc, Image<T2> &imsift,const vector<int> cellSizeVect,int stepSize,bool IsBoundaryIncluded,int nBins)
{
	// this parameter controls decay of the gradient energy falls into a bin
	// run SIFT_weightFunc.m to see why alpha = 9 is the best value
	int alpha = 9; 

	int width=imsrc.width(),height=imsrc.height(),nchannels =imsrc.nchannels();
	int nPixels = width*height;
	DImage imdx,imdy;
	// compute the derivatives;
	imsrc.dx(imdx,true);
	imsrc.dy(imdy,true);

	// get the maximum gradient over the channels and estimate the normalized gradient
	DImage magsrc(width,height,nchannels),mag(width,height),gradient(width,height,2);
	double Max;
	for(int i=0;i<nPixels;i++)
	{
		int offset = i*nchannels;
		for(int j = 0;j<nchannels;j++)
			magsrc.pData[offset+j] = sqrt(imdx.pData[offset+j]*imdx.pData[offset+j]+imdy.pData[offset+j]*imdy.pData[offset+j]);
		Max = magsrc.pData[offset];
		if(Max!=0)
		{
			gradient.pData[i*2] = imdx.pData[offset]/Max;
			gradient.pData[i*2+1] = imdy.pData[offset]/Max;
		}		
		for(int j = 1;j<nchannels;j++)
		{
			if(magsrc.pData[offset+j]>Max)
			{
				Max = magsrc.pData[offset+j];
				gradient.pData[i*2] = imdx.pData[offset+j]/Max;
				gradient.pData[i*2+1] = imdy.pData[offset+j]/Max;
			}
		}
		mag.pData[i] = Max;
	}

	// get the pixel-wise energy for each orientation band
	DImage imband(width,height,nBins);
	double theta = M_PI*2/nBins;
	double _cos,_sin,temp;
	for(int k = 0;k<nBins;k++)
	{
		_sin    = sin(theta*k);
		_cos   = cos(theta*k);
		for(int i = 0;i<nPixels; i++)
		{
			temp = __max(gradient.pData[i*2]*_cos + gradient.pData[i*2+1]*_sin,0);
			if(alpha>1)
				temp = pow(temp,alpha);
			imband.pData[i*nBins+k] = temp*mag.pData[i]; 
		}
	}
	// get the maximum cell size
	int maxCellSize = cellSizeVect[0];
	int nScales = cellSizeVect.size();
	for(int h=1;h<nScales;h++)
		maxCellSize = __max(maxCellSize,cellSizeVect[h]);

	// allocate buffer for the sift image
	int siftdim = nBins*16;
	int sift_width,sift_height,x_shift=0,y_shift=0;

	sift_width = width/stepSize;
	sift_height = height/stepSize;

	if(IsBoundaryIncluded == false)
	{
		sift_width = (width-4*maxCellSize)/stepSize;
		sift_height= (height-4*maxCellSize)/stepSize;
		x_shift = 2*maxCellSize;
		y_shift = 2*maxCellSize;
	}

	if(!imsift.matchDimension(sift_width,sift_height,siftdim*nScales))
		imsift.allocate(sift_width,sift_height,siftdim*nScales);

	for(int h=0;h<nScales;h++)
	{
		int cellSize = cellSizeVect[h];
		if(cellSize<=0)
		{
			cout<<"The cell size must be positive!"<<endl;
			return;
		}
		// filter out the SIFT feature
		DImage filter(cellSize*2+1,1);
		filter[0] = filter[cellSize+1] = 0.25;
		for(int i = 1;i<cellSize+1;i++)
			filter[i+1] = 1;
		for(int i = cellSize+2;i<cellSize*2+1;i++)
			filter[i] = 0;

		DImage imband_cell;
		imband.imfilter_hv(imband_cell,filter.data(),cellSize,filter.data(),cellSize);

		// now sample to get SIFT image
		DImage sift_cell(siftdim,1);
		for(int i=0;i<sift_height;i++)
			for(int j =0;j<sift_width;j++)
			{
				int count = 0;
				for(int ii = -1;ii<=2;ii++)
					for(int jj=-1;jj<=2;jj++)
					{
						int y = __min(__max(y_shift+i*stepSize+ii*cellSize,0),height-1);
						int x = __min(__max(x_shift+j*stepSize+jj*cellSize,0),width-1);
						
						// the following code is the same as the above two for debugging purposes
						//int y = y_shift+i*stepSize+ii*cellSize;
						//int x = x_shift+j*stepSize+jj*cellSize;
						//if (x<0 || x>=width)
						//	x = __min(__max(x,0),width-1);
						//if (y<0 || y>=height)
						//	y= __min(__max(y,0),height-1);

						memcpy(sift_cell.pData+count*nBins,imband_cell.pData+(y*width+x)*nBins,sizeof(double)*nBins);
						count++;
					}
				// normalize the SIFT descriptor
				double mag = sqrt(sift_cell.norm2());
				int offset = (i*sift_width+j)*siftdim*nScales+h*siftdim;
				//memcpy(imsift.pData+offset,sift_cell.pData,sizeof(double)*siftdim);
				for(int k = 0;k<siftdim;k++)
					imsift.pData[offset+k] = (unsigned char)__min(sift_cell.pData[k]/(mag+0.01)*255,255);//(unsigned char) __min(sift_cell.pData[k]/mag*512,255);
			}//*/
	}
}