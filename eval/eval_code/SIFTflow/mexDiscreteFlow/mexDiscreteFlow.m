% mex function to perform SIFT flow matching
%
% Usage:
%
% [flow,energylist] = mexDiscreteFlow(im1,im2,para);
% [flow,energylist] = mexDiscreteFlow(im1,im2,para,offsetX,offsetY,winSizeX,winSizeY);
%
% ** Input arguments **
%
% Arguments "im1" and "im2" are SIFT images. SIFT flow is estimated from im1 to im2. Normally 
% MATLAB function dense_sift is used to generate a dense SIFT image from a RGB image. The 
% argument "para" is the a vector specifying the parameters of matching. Let
%
%   para = [alpha,d,gamma,nIterations,nHierarchy,wsize];
%
% where each element is defined as (the number in the parenthesis is the defaut value)
%
%       alpha:        (0.01) the weight of the truncated L1-norm regularization on the flow
%       d:            (1) the threshold of the truncation
%       gamma:        (0.001) the weight of the magnitude of the flow
%       nIterations:  (40) the number of iterations
%       nHierarchy:   (2)  the number of hierarchies in the efficient BP implementation
%       wsize:        (5)  the half size of the search window 
%
% Notice that the two images are NOT required to have the same dimension. But certainly you
% wouldn't expect to get anything meaningful if their dimensions are too different.
%
% Arguments "offsetX" and "offsetY" should have the same dimension (height and width) as im1.
% These two parameters are introduced to handle coarse-to-fine SIFT flow matching. We can 
% specify the center of the matching window by defining offsetX and offsetY. When these two
% arguments are not specified, they are treated as 0.
%
% Arugments "winSizeX" and "winSizeY" should have also the same dimeiosn as im1. These two 
% parameters are introduced to overwrite "wsize" in "para", to specify pixel-wise window size.
%
% ** Output arguments **
% 
% The first argument "flow" is a [h x w x 2] matrix where flow(:,:,1) is the horizontal component
% and flow(:,:,2) is the vertical component. The height and weight of flow is the same as im1.
%
% The second argument "energylist" records the energy obtained by the BP algorithm at each 
% iteration.
%
% Ce Liu
% CSAIL, MIT
% Jan, 2009
% All rights reserved
