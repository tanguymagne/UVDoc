% The wrapper of mex function mexDenseSIFT.m. See demo_mexDenseSIFT.m for an example of usage.
%
% sift = mexDenseSIFT(im,cellSize,stepSize,IsBoundary);
%
% Input arguments
%   im  --          an RGB or grayscale image (either uint8 or double) 
%   cellSize -- (default: 3) a vector of cell size. If a scale is input, then the SIFT descriptor of one cell size is
%                        computed; if a vector is input, then multi-scale SIFT is computed and concatenated. 
%   stepSize -- (default: 1) a scale of step size in sampling image grid. If an integer larger than 1 is input, then
%                       sparse sampling of image grid is performed
%  IsBoundary-- (default: true) a boolean variable indicating whether boundary is included.
%
% Output
%   sift --         an image (with multiple channels, typicallly 128) of datatype UINT8 despite the type of the input.
%                       The maximum element wise value of sift is 255. This datatype is consistent with the byte-based
%                       SIFT flow algorithm
%
%  Ce Liu
%  June 29, 2010