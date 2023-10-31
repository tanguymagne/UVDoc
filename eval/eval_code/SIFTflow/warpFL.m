% warp i2 according to flow field in vx vy
function [warpI2,I]=warp(i2,vx,vy)
[M,N]=size(i2);
[x,y]=meshgrid(1:N,1:M);
warpI2=interp2(x,y,i2,x+vx,y+vy,'bicubic');
I=find(isnan(warpI2));
warpI2(I)=zeros(size(I));
