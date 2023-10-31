function [vx,vy] = siftFlow(im1,im2)
    %UNTITLED3 Summary of this function goes here
    %   Detailed explanation goes here
    im1=imresize(imfilter(im1,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
    im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
    
    im1=im2double(im1);
    im2=im2double(im2);
    
    cellsize=3;
    gridspacing=1;
    
    sift1 = mexDenseSIFT(im1,cellsize,gridspacing);
    sift2 = mexDenseSIFT(im2,cellsize,gridspacing);
    
    SIFTflowpara.alpha=2*255;
    SIFTflowpara.d=40*255;
    SIFTflowpara.gamma=0.005*255;
    SIFTflowpara.nlevels=4;
    SIFTflowpara.wsize=2;
    SIFTflowpara.topwsize=10;
    SIFTflowpara.nTopIterations = 60;
    SIFTflowpara.nIterations= 30;
    
    
    [vx,vy,~]=SIFTflowc2f(sift1,sift2,SIFTflowpara);
end
    
    