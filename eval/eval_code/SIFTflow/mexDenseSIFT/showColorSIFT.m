function SIFTcolor=showColorSIFT(SIFT)
%
% It reduces the dimensionality of SIFT to 3 dimensions and outputs a color
% coded image
%
% It will map the first PCA dimension to luminance (R+G+B), then it will
% map the second to R-G and the third one to (R+G)/2-B

load ('pcSIFT', 'pcSIFT')

[nrows ncols nfeatures] = size(SIFT);
SIFTpca = pcSIFT(:,1:3)'*double(reshape(SIFT, [nrows*ncols nfeatures]))';

A = inv([1 1 1; 1 -1 0; .5 .5 -1]);
%A = eye(3,3);

SIFTcolor = A * SIFTpca(1:3,:);
SIFTcolor = reshape(SIFTcolor', [nrows ncols 3]);
SIFTcolor = SIFTcolor - min(SIFTcolor(:));
SIFTcolor = SIFTcolor / max(SIFTcolor(:));
%SIFTcolor = uint8(255*SIFTcolor / max(SIFTcolor(:)));
%imshow(SIFTcolor)

