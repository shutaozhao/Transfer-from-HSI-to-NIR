%% ================== Repfile_EasyTL_six ===========================
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
        xsrc_new = []; xtar_new = [];
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
        
       %% ================= Repfile transformation ==================
        t1 = clock;
        % compute the averaged differences in each class (here,we have six classes)
        D = tar_clb - src_clb;
        num1 = size(D,1);
        num2 = num1/6;
        D_m = [];
        for nn = 1:num2:num1
            D_mean = mean(D(nn:(nn+(num2-1)),:),1);
            D_m = [D_m;D_mean];
        end
        % Use the averaged spectral differences of standard samples to transform all dataset
        for mm = 1:75            
            xsrc_new(mm,:) = xsrc_new(mm,:)+D_m(1,:);
            xsrc_new(75+mm,:) = xsrc_new(75+mm,:)+D_m(2,:);    
            xsrc_new(150+mm,:) = xsrc_new(150+mm,:)+D_m(3,:);
            xsrc_new(225+mm,:) = xsrc_new(225+mm,:)+D_m(4,:);
            xsrc_new(300+mm,:) = xsrc_new(300+mm,:)+D_m(5,:);
            xsrc_new(375+mm,:) = xsrc_new(375+mm,:)+D_m(6,:);
        end
        
       %% ================== EasyTL ======================
        t1 = clock;
        Xs = xsrc_new;    Xt = xtar;
        Ys = y;       Yt = y;
        intra_align = 'coral';
        dist = 'euclidean';
        lp = 'linear';
        [acc,y_pred] = EasyTL(Xs,Ys,Xt,Yt,intra_align,dist,lp);
        Acc_task1_tar = [Acc_task1_tar,acc]; clear acc
        task1_predict_label = [task1_predict_label,y_pred];   clear y_pred
        t2 = clock;
        time = etime(t2,t1);
        Task1_running_time = [Task1_running_time,time];  clear time
    end
end    
