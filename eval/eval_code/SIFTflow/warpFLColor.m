% Function to warp color image im2 to the grid of im1. It uses the pixels
% in im1 to fill in the holes of warpI2 if there is any in the warping
function warpI2=warpFLColor(im1,im2,vx,vy)
if isfloat(im1)~=1
    im1=im2double(im1);
end
if isfloat(im2)~=1
    im2=im2double(im2);
end
if exist('vy')~=1
    vy=vx(:,:,2);
    vx=vx(:,:,1);
end
nChannels=size(im1,3);
for i=1:nChannels
    [im,isNan]=warpFL(im2(:,:,i),vx,vy);
    temp=im1(:,:,i);
    im(isNan)=temp(isNan);
    warpI2(:,:,i)=im;
end