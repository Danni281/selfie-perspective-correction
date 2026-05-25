%
%% LICENSE
% 
% Copyright 2014:
%   - Xavier P.Burgos-Artizzu [xavier.burgos@technicolor.com]
%   - Matteo Ruggero Ronchi   [mronchi@caltech.edu]
%   - Pietro Perona           [perona@caltech.edu]
% Please email us if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%% DESCRIPTION
%
% Version [1.0.0]
%
% This script configures the dataset paths and visualizes ground truth
% annotations for the three different face landmarks vectors collected:
% - MANUAL:    55 Facial Landmarks (Custom format) annotated by humans
% - RCPR 300F: 68 Facial Landmarks (Multi-Pie format) automatically 
%              annotated by RCPR
% - RCPR CMDP: 55 Facial Landmarks (Custom format) automatically annotated 
%              by RCPR

clear;clc;

%% CONFIGURE PATH VARIABLES

DINFO_PATH = './CMDP-ANNO/';
DATASET_BASE_DIR_PATH = './CMDP';

%% LOAD DATASET INFO AND SET PATHS
%
% NOTE:    If you are working on a WINDOWS OS you might need to change the
% code in lines 43 -> 50 so that the path separator is '\' and not '/'.

load([DINFO_PATH 'dinfo.mat']);

if CONFIG_FLAG == 0,

    for i = 1:numSubjects, 
        infos{i}.baseDir = DATASET_BASE_DIR_PATH;
    end

    for i = 1:numSubjects, 
        for j = 1:numPics,
            
            str = imagePathsOriginal{i,j};
            tkns = strsplit(str,'/');
            str = [DATASET_BASE_DIR_PATH '/' tkns{1} '/' tkns{2}];
            imagePathsOriginal{i,j} = str;
            
            str = imagePaths_Manual_ST{i,j};
            tkns = strsplit(str,'/');
            str = [DATASET_BASE_DIR_PATH '/' tkns{1} '/' tkns{2} '/' tkns{3}];
            imagePaths_Manual_ST{i,j} = str;
            
            if ~isempty(find(testSub == i, 1))
                str = imagePaths_RcprCMDP_ST{i,j};
                tkns = strsplit(str,'/');
                str = [DATASET_BASE_DIR_PATH '/' tkns{1} '/' tkns{2} '/' tkns{3}];
                imagePaths_RcprCMDP_ST{i,j} = str;
            end
            
            str = imagePaths_Rcpr300F_ST{i,j};
            tkns = strsplit(str,'/');
            str = [DATASET_BASE_DIR_PATH '/' tkns{1} '/' tkns{2} '/' tkns{3}];
            imagePaths_Rcpr300F_ST{i,j} = str;
            
        end
    end

    CONFIG_FLAG = 1;
    
    save([DINFO_PATH 'dinfo.mat'],'CONFIG_FLAG', ...
        'distancesVec_FT', 'distancesVec_MM', ...
        'fids_Manual_OR','fids_Manual_ST','fids_Rcpr300F_OR', ...
        'fids_Rcpr300F_ST', 'fids_RcprCMDP_OR', 'fids_RcprCMDP_ST', ...
        'imagePathsOriginal','imagePaths_Manual_ST', ...
        'imagePaths_RcprCMDP_ST','imagePaths_Rcpr300F_ST',...
        'infos','numPics','numSubjects','testSub','trainSub');

end
    
%% VISUALIZE GROUNDTRUTH ANNOTATIONS
%
% NOTE:    The RcprCMDP annotations are currently provided only for the 
% subjects in the test set. The indices of the subjects in the test set are
% contained in the vector testSub contained in the dinfo.mat file. 
% Soon we will upload RcprCMDP annotations for all subjects.

for i=1:numSubjects
    
    infos{i}
    
    for j=1:numPics
        
        im1 = imread(imagePathsOriginal{i,j});
        
        im4 = imread(imagePaths_Manual_ST{i,j});
        if ~isempty(find(testSub == i, 1))
            im5 = imread(imagePaths_RcprCMDP_ST{i,j});
        end
        im6 = imread(imagePaths_Rcpr300F_ST{i,j});
        
        
        clf;
        
        subplot(2,3,1),imshow(im1); hold on;
        scatter(fids_Manual_OR(i,1:55,j),fids_Manual_OR(i,56:110,j),100,'r','fill');
        title('Manual - OR')
        
        subplot(2,3,2)
        if ~isempty(find(testSub == i, 1))
            imshow(im1); hold on;
            scatter(fids_RcprCMDP_OR(i,1:55,j),fids_RcprCMDP_OR(i,56:110,j),100,'r','fill');
            title('RCPR CMDP - OR')
        else
            title(sprintf('RCPR CMDP NOT AVAILABLE\nFOR SUBJECTS IN TRAIN SET'))
        end
        
        subplot(2,3,3),imshow(im1); hold on;
        scatter(fids_Rcpr300F_OR(i,1:68,j),fids_Rcpr300F_OR(i,69:136,j),100,'r','fill');
        title('RCPR 300F - OR')
        
        subplot(2,3,4),imshow(im4); hold on;
        scatter(fids_Manual_ST(i,1:55,j),fids_Manual_ST(i,56:110,j),100,'r','fill');
        title('Manual - ST')
        
        subplot(2,3,5)
        if ~isempty(find(testSub == i, 1))
            imshow(im5); hold on;
            scatter(fids_RcprCMDP_ST(i,1:55,j),fids_RcprCMDP_ST(i,56:110,j),100,'r','fill');
            title('RCPR CMDP - ST')
        else
            title(sprintf('RCPR CMDP NOT AVAILABLE\nFOR SUBJECTS IN TRAIN SET'))
        end
        
        subplot(2,3,6),imshow(im6); hold on;
        scatter(fids_Rcpr300F_ST(i,1:68,j),fids_Rcpr300F_ST(i,69:136,j),100,'r','fill');
        title('RCPR 300F - ST')
        
        pause(.05)
    end
end
