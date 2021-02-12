%% ========================= DS_SVM =============================== 
clc;clear all;close all;

% y is the label of dataset
y = csvread('E:\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\label.csv');

HSI_datasets = {'HSI_s1','HSI_s2','HSI_s3'};
NIR_datasets = {'NIR_s1','NIR_s2','NIR_s3'};

%% ------------------ HSI transfer to NIR ------------------->
task1_acc_train = []; task1_acc_tar = []; 
Best_C_task1 = []; Best_G_task1 = []; Task1_running_time = [];
MMD_linear_HN05task3 = [];  MMD_rbf_HN05task3 = [];   
task1_tar_predict_label = [];
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
       %% === divide the standard samples (src_clb and tar_clb) in the ratio of 0.5
        ratio = 0.5;    
        % ratio is also set to be 0.6 / 0.7 / 0.8 / 0.9
        % here, take 0.5 as an example 
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
        
       %% Divide 20% dataset in the source domain for parameter optimization
        ratio = 0.8;    
        X = xsrc;
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
        
       %% =============== Compute MMD ===================
        X = xsrc; Y = tar_ds;
        kernel_type = 'linear';
        gamma = 1;
        mmd_linear_05task3 =  mmd_matlab(X, Y, kernel_type, gamma)
        MMD_linear_HN05task3 = [MMD_linear_HN05task3, mmd_linear_05task3];
        kernel_type = 'rbf';
        gamma = bestg;
        mmd_rbf_05task3 =  mmd_matlab(X, Y, kernel_type, gamma)
        MMD_rbf_HN05task3 = [MMD_rbf_HN05task3, mmd_rbf_05task3];    % save MMD
        
        %% ------ train SVM model by train set in source domain -------
        train_group = src_train_lbl; train = src_train;
        model = svmtrain(train_group,train,cmd);     
        
        %% ------ predict the pollution class of target domain --------
        test_group = y;  test = tar_ds;
        [predict_label, accuracy, dec_values]=svmpredict(test_group,test,model);
        acc = accuracy(1,1);
        task1_acc_tar = [task1_acc_tar,acc];    % save accuracy
        task1_tar_predict_label = [task1_tar_predict_label,predict_label];  % save predicted_class
        clear predict_label

        t2 = clock;               % end the timer
        time = etime(t2,t1);
        Task1_running_time = [Task1_running_time,time];    % save running_time
    end
end    

%% ================= Save Results ======================= 
title1 = {'Task1_acc_tar'};
results_acc = [task1_acc_tar'];
A = [title1;num2cell(results_acc)];

title2 = {'Task1_Best_C','Task1_Best_G'};
results_parameters = [Best_C_task1',Best_G_task1'];
B = [title2;num2cell(results_parameters)];

title3 = {'Task1_running_time'};
results_running_time = [Task1_running_time'];
C = [title3;num2cell(results_running_time)];

title4 = {'Task1_MMD_linear','Task1_MMD_rbf'};
results_MMD = [MMD_linear_HN05task3',MMD_rbf_HN05task3'];
D = [title4;num2cell(results_MMD)];

E = task1_tar_predict_label;

xlswrite('E:\Experiment\HSI_transfer_NIR\Results_Revise\DS_SVM\DS0.5_acc.xlsx',A,'sheet1');
xlswrite('E:\Experiment\HSI_transfer_NIR\Results_Revise\DS_SVM\DS0.5_parameters.xlsx',B,'sheet1');
xlswrite('E:\Experiment\HSI_transfer_NIR\Results_Revise\DS_SVM\DS0.5_running_time.xlsx',C,'sheet1');
xlswrite('E:\Experiment\HSI_transfer_NIR\Results_Revise\DS_SVM\DS0.5_MMD.xlsx',D,'sheet1');
xlswrite('E:\Experiment\HSI_transfer_NIR\Results_Revise\DS_SVM\DS0.5_tar_predict_value.xlsx',E,'sheet1');
