% function to do coarse to fine SIFT flow matching
function [vx,vy,energylist]=SIFTflowc2f(im1,im2,SIFTflowpara,isdisplay,Segmentation)

if isfield(SIFTflowpara,'alpha')
    alpha=SIFTflowpara.alpha;
else
    alpha=0.01;
end

if isfield(SIFTflowpara,'d')
    d=SIFTflowpara.d;
else
    d=alpha*20;
end

if isfield(SIFTflowpara,'gamma')
    gamma=SIFTflowpara.gamma;
else
    gamma=0.001;
end

if isfield(SIFTflowpara,'nlevels')
    nlevels=SIFTflowpara.nlevels;
else
    nlevels=4;
end

if isfield(SIFTflowpara,'wsize')
    wsize=SIFTflowpara.wsize;
else
    wsize=3;
end

if isfield(SIFTflowpara,'topwsize')
    topwsize=SIFTflowpara.topwsize;
else
    topwsize=10;
end

if isfield(SIFTflowpara,'nIterations')
    nIterations=SIFTflowpara.nIterations;
else
    nIterations=40;
end

if isfield(SIFTflowpara,'nTopIterations')
    nTopIterations=SIFTflowpara.nTopIterations;
else
    nTopIterations=100;
end

if exist('isdisplay','var')~=1
    isdisplay=false;
end

if exist('Segmentation','var')==1
    IsSegmentation=true;
else
    IsSegmentation=false;
end

% build the pyramid
pyrd(1).im1=im1;
pyrd(1).im2=im2;
if IsSegmentation
    pyrd(1).seg=Segmentation;
end

for i=2:nlevels
    pyrd(i).im1=imresize(imfilter(pyrd(i-1).im1,fspecial('gaussian',5,0.67),'same','replicate'),0.5,'bicubic');
    pyrd(i).im2=imresize(imfilter(pyrd(i-1).im2,fspecial('gaussian',5,0.67),'same','replicate'),0.5,'bicubic');
%     pyrd(i).im1 = reduceImage(pyrd(i-1).im1);
%     pyrd(i).im2 = reduceImage(pyrd(i-1).im2);
    if IsSegmentation
        pyrd(i).seg=imresize(pyrd(i-1).seg,0.5,'nearest');
    end
end

for i=1:nlevels
    [height,width,nchannels]=size(pyrd(i).im1);
    [height2,width2,nchannels]=size(pyrd(i).im2);
    [xx,yy]=meshgrid(1:width,1:height);    
    pyrd(i).xx=round((xx-1)*(width2-1)/(width-1)+1-xx);
    pyrd(i).yy=round((yy-1)*(height2-1)/(height-1)+1-yy);
end

nIterationArray=round(linspace(nIterations,nIterations,nlevels));

for i=nlevels:-1:1
    if isdisplay
        fprintf('Level: %d...',i);
    end
    [height,width,nchannels]=size(pyrd(i).im1);
    [height2,width2,nchannels]=size(pyrd(i).im2);
    [xx,yy]=meshgrid(1:width,1:height);
    
    if i==nlevels
%         vx=zeros(height,width);
%         vy=vx;
        vx=pyrd(i).xx;
        vy=pyrd(i).yy;
        
        winSizeX=ones(height,width)*topwsize;
        winSizeY=ones(height,width)*topwsize;
    else
%         vx=imresize(vx-pyrd(i+1).xx,[height,width],'bicubic')*2+pyrd(i).xx;
%         vy=imresize(vy-pyrd(i+1).yy,[height,width],'bicubic')*2+pyrd(i).yy;
        
%         winSizeX=decideWinSize(vx,wsize);
%         winSizeY=decideWinSize(vy,wsize);
        vx=round(pyrd(i).xx+imresize(vx-pyrd(i+1).xx,[height,width],'bicubic')*2);
        vy=round(pyrd(i).yy+imresize(vy-pyrd(i+1).yy,[height,width],'bicubic')*2);
        
        winSizeX=ones(height,width)*(wsize+i-1);
        winSizeY=ones(height,width)*(wsize+i-1);
    end
    if nchannels<=3
        Im1=im2feature(pyrd(i).im1);
        Im2=im2feature(pyrd(i).im2);
    else
        Im1=pyrd(i).im1;
        Im2=pyrd(i).im2;
    end
    % compute the image-based coefficient
    if IsSegmentation
        imdiff=zeros(height,width,2);
        imdiff(:,1:end-1,1)=double(pyrd(i).seg(:,1:end-1)==pyrd(i).seg(:,2:end));
        imdiff(1:end-1,:,2)=double(pyrd(i).seg(1:end-1,:)==pyrd(i).seg(2:end,:));
        Im_s=imdiff*alpha+(1-imdiff)*alpha*0.01;
        Im_d=imdiff*alpha*100+(1-imdiff)*alpha*0.01*20;
    end
    if i==nlevels
        if IsSegmentation
            [flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nTopIterations,2,topwsize],vx,vy,winSizeX,winSizeY,Im_s,Im_d);
        else
            [flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nTopIterations,2,topwsize],vx,vy,winSizeX,winSizeY);
            %[flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nTopIterations,0,topwsize],vx,vy,winSizeX,winSizeY);
        end
%         [flow1,foo1]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterationArray(i),0,topwsize],vx,vy,winSizeX,winSizeY);
%         [flow2,foo2]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nTopIterations,2,topwsize],vx,vy,winSizeX,winSizeY);
%         if foo1(end)<foo2(end)
%             flow=flow1;
%             foo=foo1;
%         else
%             flow=flow2;
%             foo=foo2;
%         end
    else
        %[flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterations,nlevels-i,wsize],vx,vy,winSizeX,winSizeY);
        if IsSegmentation
            [flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterationArray(i),nlevels-i,wsize],vx,vy,winSizeX,winSizeY,Im_s,Im_d);
        else
            [flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterationArray(i),nlevels-i,wsize],vx,vy,winSizeX,winSizeY);
            %[flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterationArray(i),0,wsize],vx,vy,winSizeX,winSizeY);
        end
%         [flow,foo]=mexDiscreteFlow(Im1,Im2,[alpha,d,gamma*2^(i-1),nIterationArray(i),0,wsize],vx,vy,winSizeX,winSizeY);
    end
    energylist(i).data=foo;
    vx=flow(:,:,1);
    vy=flow(:,:,2);
    if isdisplay
        fprintf('done!\n');
    end
end

function winSizeIm=decideWinSize(offset,wsize)

% design the DOG filter
f1=fspecial('gaussian',9,1);
f2=fspecial('gaussian',9,.5);
f=f2-f1;

foo=imfilter(abs(imfilter(offset,f,'same','replicate')),fspecial('gaussian',9,1.5),'same','replicate');

Min=min(foo(:));
Max=max(foo(:));
winSizeIm=wsize*(foo-Min)/(Max-Min)+wsize;



