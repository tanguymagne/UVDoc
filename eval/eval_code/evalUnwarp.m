function [ms, ld, lid] = evalUnwarp(x, y, vx, vy)
%EVALUNWARP compute MSSSIM, LD and LiD between the unwarped image and the scan
%   A:      unwarped image
%   ref:    reference image, the scan image
%   ms:     returned MS-SSIM value
%   ld:     returned local distortion value
%   Matlab image processing toolbox is necessary to compute ssim. The weights 
%   for multi-scale ssim is directly adopted from:
%
%   Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural 
%   similarity for image quality assessment." In Signals, Systems and Computers, 
%   2004. Conference Record of the Thirty-Seventh Asilomar Conference on, 2003. 
%
%   Local distortion relies on the paper:
%   Liu, Ce, Jenny Yuen, and Antonio Torralba. "Sift flow: Dense correspondence 
%   across scenes and its applications." In PAMI, 2010.
%
%   and its implementation:
%   https://people.csail.mit.edu/celiu/SIFTflow/

% Compute LD
d = sqrt(vx.^2 + vy.^2);
ld = mean(d(:));

% Compute MS-SSIM
wt = [0.0448 0.2856 0.3001 0.2363 0.1333];
ss = zeros(5, 1);
for s = 1 : 5
    ss(s) = ssim(x, y);
    x = impyramid(x, 'reduce');
    y = impyramid(y, 'reduce');
end
ms = wt * ss;

end