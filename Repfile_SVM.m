%% ====================== Repfile_SVM ===========================
clc;clear all;close all;
% y is the label of dataset
y = csvread('E:\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\label.csv');

HSI_datasets = {'HSI_s1','HSI_s2','HSI_s3'};
NIR_datasets = {'NIR_s1','NIR_s2','NIR_s3'};

%% ------------------ HSI transfer to NIR ------------------->
task1_acc_train = []; task1_acc_tar = []; 
Best_C_task1 = []; Best_G_task1 = []; Task1_running_time = [];
MMD_linear_HN06task3 = [];  MMD_rbf_HN06task3 = [];   

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
        xsrc_new = [];
       %% === divide the standard samples (src_clb and tar_clb) in the ratio of 0.5
        ratio = 0.6;    
        % ratio is also set to be 0.5 / 0.7 / 0.8 / 0.9
        % here, take 0.6 as an example 
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
        
       %% Divide 20% dataset in the source domain for parameter optimization
        ratio = 0.8;    
        X = xsrc_new;
        [X_train, y_train, X_test, y_test] = split_train_test(X, y, ratio);
        src_train = X_train; src_train_lbl = y_train;
        src_vld = X_test;   src_vld_lbl = y_test;
        clear X_train  y_train  X_test  y_test        

       %% ======= find the best hyperparameters ========
        train_label = src_vld_lbl;
        train = src_vld;
        [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,-10,10,-15,10,3,0.1,0.1,4.5);  
        Best_C_task1 = [Best_C_task1,bestc];
        Best_G_task1 = [Best_G_task1,bestg];
        cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
        
       % =============== Compute MMD ===================
        X = xsrc_new; Y = xtar;
        kernel_type = 'linear';
        gamma = 1;
        mmd_linear_06task3 =  mmd_matlab(X, Y, kernel_type, gamma)
        MMD_linear_HN06task3 = [MMD_linear_HN06task3, mmd_linear_06task3];
        kernel_type = 'rbf';
        gamma = bestg;
        mmd_rbf_06task3 =  mmd_matlab(X, Y, kernel_type, gamma)
        MMD_rbf_HN06task3 = [MMD_rbf_HN06task3, mmd_rbf_06task3];
        
       %% ------ train SVM model by train set in source domain -------
        train_group = src_train_lbl; train = src_train;
        model = svmtrain(train_group,train,cmd);

       %% ------ predict the pollution class of target domain --------
        test_group = y;  test = xtar;
        [predict_label, accuracy, dec_values]=svmpredict(test_group,test,model);
        acc = accuracy(1,1);
        task1_acc_tar = [task1_acc_tar,acc];

        t2 = clock;
        time = etime(t2,t1);
        Task1_running_time = [Task1_running_time,time];
    end
end    
