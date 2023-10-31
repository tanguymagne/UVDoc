The *.cpp and *.h files in this folder are used to mex a matlab dll file for computing dense SIFT features. Use the following command to compile in Matlab (tested on version 7.6 or later)

mex mexDenseSIFT.cpp Matrix.cpp Vector.cpp

It has been tested in Windows 7 x64, Linux x64 and Mac OS 10.6. Precompiled versions of Windows 7 x64 and Mac OS 10.6 are included in the folder.


------------------------- Important -------------------------

You must change one line in order to compile correctly. On line 5 of project.h, you should comment this line if you are compiling using visual studio in windows, or uncomment if you are in linux or mac os.

-------------------------------------------------------------