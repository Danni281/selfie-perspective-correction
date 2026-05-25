% Copyright (c) 2014:
%   - Xavier P.Burgos-Artizzu [xavier.burgos@technicolor.com]
%   - Matteo Ruggero Ronchi   [mronchi@caltech.edu]
%   - Pietro Perona           [perona@caltech.edu]
% Please email us if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]

CONTENT OF THIS DIRECTORY:

- dinfo.mat: 

MATLAB variable containing: 
1) the groundtruth annotations for the three methods used (MANUAL, RCPR_300F, RCPR_CMDP) for 51 subjects and 7 distances; 
2) paths of the original and standardized versions of the images in the CMDP Dataset; 
3) subject info form the CMDP Dataset (gender, occlusions, ...); 
4) CMDP Dataset info (number of subjects and pics, indices of subjects used in the training and test set for our routines).

This is the variable that you'll have to load in order to replicate all our code results.

- show_DINFO.m:

MATLB script visualizing the annotations for all the images in the dataset.

- bsd.txt:

License information for this code and the CMDP dataset.

INSTRUCTIONS TO VIEW ANNOTATIONS:

1) Download CMDP_1.zip and CMDP_2.zip
2) Unzip the two files and move their contents under a unique directory
3) Open MATLAB file 'show_DINFO.m'
4) Configure PATHS of the 'dinfo.mat' file downloaded and the dataset directory
5) Run 
