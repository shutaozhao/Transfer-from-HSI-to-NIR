%% ================== DS_EasyTL_six ===========================
clc;clear all;close all;
y = csvread('E:\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\label.csv');
HSI_datasets = {'HSI_s1','HSI_s2','HSI_s3'};
NIR_datasets = {'NIR_s1','NIR_s2','NIR_s3'};

%% ------------------ HSI transfer to NIR ------------------->
Acc_task1_tar = [];    Task1_running_time = [];   task1_predict_label = [];
for i = 1:3
   %% =========================== Read datasets ===================================
    data_src = HSI_datasets{i};
    data_tar = NIR_datasets{i};
    Xsrc = cell2mat(struct2cell(load(['E:\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\' data_src '.mat'])));
    Xtar = cell2mat(struct2cell(load(['E:\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\' data_tar '.mat'])));
   
    %% ========== Preprocess (Area_normalization and Zscore normalization) ==========
    fts = Xsrc;
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));  
    xsrc = zscore(fts,1);    clear fts

    fts = Xtar;
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));  
    xtar = zscore(fts,1);    clear fts
    
    for j = 1:30
        % Repeat 30 times (you can also run only one time)
        tar_ds = []; 
       %% === divide the standard samples (src_clb and tar_clb) in the ratio of 0.9 
        % ratio is also set to be 0.5 / 0.6 / 0.7 / 0.8 
        % here, take 0.9 as an example 
        ratio = 0.9;    
        X = xsrc;
        [X_train, y_train, X_test, y_test] = split_train_test(X, y, ratio);
        src_clb = X_test;   src_clb_lbl = y_test;
        clear X_train  y_train  X_test  y_test
        X = xtar;        
        [X_train, y_train, X_test, y_test] = split_train_test(X, y, ratio);
        tar_clb = X_test;   tar_clb_lbl = y_test;
        clear X_train  y_train  X_test  y_test
        
       %% ============= DS transformation =============
        % Input: n_samples * dim
        % P is the generalized inverse matrix of the standard sample in target domain 
        % F is the transformation coefficient matrix
        % Use F to correct all samples in the target domain
        t1 = clock;  % start the timer

        P = pinv(tar_clb);
        F = P * src_clb;
        tar_ds = xtar * F;  
        
       %% ================== EasyTL ======================    
        Xs = xsrc;    Xt = tar_ds;
        Ys = y;       Yt = y;
        
        intra_align = 'coral';
        dist = 'euclidean';
        lp = 'linear';
        [acc,y_pred] = EasyTL(Xs,Ys,Xt,Yt,intra_align,dist,lp);
        Acc_task1_tar = [Acc_task1_tar,acc];
        task1_predict_label = [task1_predict_label,y_pred];   clear y_pred
        
        t2 = clock;
        time = etime(t2,t1);
        Task1_running_time = [Task1_running_time,time]; clear time;
    end
end    
