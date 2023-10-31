#include <QtGui/QApplication>
#include <QtCore/QCoreApplication>
#include "Image.h"
#include "BPFlow.h"
#include "stdio.h"
#include "time.h"

int main(int argc, char *argv[])
{
	time_t start,end;
	 double dif;

	QCoreApplication a(argc, argv);

	time (&start);

	BiImage im1,im2,Im1,Im2;
	if(!im1.imread("sflowg.00.jpg"))
	{
		printf("Error in loading frame 1!");
		return -1;
	}
	if(!im2.imread("sflowg.01.jpg"))
	{
		printf("Error in loading frame 2!");
		return -1;
	}
	//if(!im1.imread("scene1.row2.jpg"))
	//{
	//	printf("Error in loading frame 1!");
	//	return -1;
	//}
	//if(!im2.imread("scene1.row3.jpg"))
	//{
	//	printf("Error in loading frame 2!");
	//	return -1;
	//}

	im1.GaussianSmoothing(Im1,.8,5);
	im2.GaussianSmoothing(Im2,.8,5);
	Im1.imresize(0.5);
	Im2.imresize(0.5);
	//Im1=im1;
	//Im2=im2;

	double alpha=0.03*255;
	double gamma=0.002*255;
	BPFlow bpflow;
	int wsize=7;
	
	bpflow.setDataTermTruncation(true);
	//bpflow.setTRW(true);
	//bpflow.setDisplay(false);
	bpflow.LoadImages(Im1.width(),Im1.height(),Im1.nchannels(),Im1.data(),Im2.data());
	bpflow.setPara(alpha*2,alpha*20);
	bpflow.setHomogeneousMRF(wsize);
	bpflow.ComputeDataTerm();
	bpflow.ComputeRangeTerm(gamma);
	bpflow.MessagePassing(100,3);

	//for(int i=0;i<55;i++)
	//{
	//	double CTRW=(i+1)*0.02;
	//	bpflow.setCTRW(CTRW);
	//	printf("No.%d CTRW=%f  energy=%f\n",i+1,CTRW,bpflow.MessagePassing(300,1));
	//}
	
	//bpflow.MessagePassing(60);
	bpflow.ComputeVelocity();

	DImage vx(Im1.width(),Im1.height()),vy(Im1.width(),Im1.height());
	for(int i=0;i<Im1.npixels();i++)
	{
		vx.data()[i]=bpflow.flow().data()[i*2];
		vy.data()[i]=bpflow.flow().data()[i*2+1];
	}
	vx.imwrite("vx_discrete.jpg",ImageIO::normalized);
	vy.imwrite("vy_discrete.jpg",ImageIO::normalized);

	time (&end);
	dif = difftime (end,start);
    printf ("It took you %.2lf seconds to run SIFT flow.\n", dif );

	//return a.exec();
	return 1;
}
