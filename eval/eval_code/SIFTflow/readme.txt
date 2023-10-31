This is the software package of our TPAMI paper:

C. Liu, J. Yuen and A. Torralba. SIFT Flow: Dense Correspondence across Scenes and its Applications. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2010.

Please cite our paper if you use our code for your research. Also, please notice that there is slight change in the package compared to the ECCV version. Obtaining dense SIFT features has been mexed.


Please run demo.h first. If error appears, please go to "mexDenseSIFT" and "mexDiscreteFlow" subfolders and follow the instructions in readme.txt (yes, there is readme.txt in each folder) to compile the cpp files. 

------------------------- Important -------------------------

You must change one line in order to compile correctly. On line 5 of project.h, you should comment this line if you are compiling using visual studio in windows, or uncomment if you are in linux or mac os.

-------------------------------------------------------------

Run demo.m in MATLAB and you will see how SIFT flow works.

Enjoy!


Ce Liu

Microsoft Research New England
Sep 2010