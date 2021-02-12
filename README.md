# Transfer-from-HSI-to-NIR
This directory contains the code for paper "Transfer learning strategy for plastic pollution detection in soil: Calibration transfer from high-throughput HSI system to NIR sensor" published in: Chemosphere.
# Requirements
all codes is written using Matlab2020a.
# Codes
'libsvm-3.1-[FarutoUltimate3.1Mcode]' and 'libsvm-3.24' are toolboxes for SVM. They should be added in the running path firstly.
'SVM', 'DS_SVM', 'Repfile_SVM', 'EasyTL_SIX', 'DS_EasyTL' and 'Repfile_EasyTL' are the main codes for the paper, which construct models and output results.
'split_train_test' is the custom function for dividing datasets.
'CORAL_map', 'EasyTL', 'getGFKDim', 'GFK_map', 'label_prop' and 'mmd_matlab' are the codes which shared by Jingdong Wang (https://github.com/jindongwang/transferlearning/tree/master/code/traditional). I have cited his work in the paper. Thanks for his outstanding work for my inspiration!
# Reference
If you find this code helpful, please cite it as:
{Shutao Zhao, Zhengjun Qiu, Yong He. Transfer learning strategy for plastic pollution detection in soil: Calibration transfer from high-throughput HSI system to NIR sensor. Chemosphere. 2021.}
