%% ====================== SVM ===========================
clc;clear all;close all;
% y is the label of dataset
y = csvread('D:\Document\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\label.csv');

HSI_datasets = {'HSI_s1','HSI_s2','HSI_s3'};
NIR_datasets = {'NIR_s1','NIR_s2','NIR_s3'};

%% --------------- HSI transfer to NIR ---------------->
task1_acc_src = []; task1_acc_tar = []; 
Best_C_task1 = [];  Best_G_task1 = [];   Task1_running_time = [];
task1_src_predict_label = [];
task1_tar_predict_label = [];
for i = 1 : 3
    %% =========================== Read datasets ===================================
    data_src = HSI_datasets{i};
    data_tar = NIR_datasets{i};
    Xsrc = cell2mat(struct2cell(load(['D:\Document\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\' data_src '.mat'])));
    Xtar = cell2mat(struct2cell(load(['D:\Document\Experiment\HSI_transfer_NIR\Test\Datasets\six_category_datasets\' data_tar '.mat'])));
    for j = 1:30
        % Repeat 30 times (you can also run only one time)
       %% ===== divide train/validation/test sets =====
        ratio = 0.6;
        X = Xsrc;    Y = y;
        [X_train, y_train, X_test, y_test] = split_train_test(X, Y, ratio);
        src_train = X_train; src_trainlbl = y_train;
        src_test = X_test;   src_testlbl = y_test;
        clear X X_train X_test y_train y_test

        ratio = 0.5;
        X = src_test; Y = src_testlbl;
        [X_train, y_train, X_test, y_test] = split_train_test(X, Y, ratio);
        src_vld = X_train;   src_vld_lbl = y_train;
        src_test = X_test;   src_testlbl = y_test;
        clear X X_train X_test y_train y_test

       %% ========== Preprocess (Area_normalization and Zscore normalization) ==========
        fts = src_train;
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));  
        [xsrc_train,mu,sig] = zscore(fts,1);     clear fts 
        fts = src_vld;
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));  
        xsrc_vld = (fts-mu)./sig;                clear fts   

        fts = Xtar;
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));  
        xtar = zscore(fts,1);    clear fts
        
       %% ============================== SVM =====================================
        t1 = clock;
       %% --- find the best hyperparameters by validation set in source domain ---
        train_label = src_vld_lbl;
        train = xsrc_vld;
        [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,-10,10,-15,10,3,0.1,0.1,4.5);  
        Best_C_task1 = [Best_C_task1,bestc];
        Best_G_task1 = [Best_G_task1,bestg];
        cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];

       %% ------------ train SVM model by train set in source domain -------------
        train_group = src_trainlbl; train = xsrc_train;
        model = svmtrain(train_group,train,cmd);

       %% ------ predict the pollution class of source domain --------
        test_group = src_testlbl;  test = xsrc_test;
        [predict_label, accuracy, dec_values]=svmpredict(test_group,test,model);
        acc_src = accuracy(1,1);   clear accuracy
        task1_acc_src = [task1_acc_src,acc_src];
        task1_src_predict_label = [task1_src_predict_label,predict_label];   clear predict_label

       %% ------ predict the pollution class of target domain --------
        test_group = y;  test = xtar;
        [predict_label, accuracy, dec_values]=svmpredict(test_group,test,model);
        acc_tar = accuracy(1,1);   clear accuracy
        task1_acc_tar = [task1_acc_tar,acc_tar];   
        task1_tar_predict_label = [task1_tar_predict_label,predict_label];   clear predict_label
        
        t2 = clock;
        time = etime(t2,t1);
        Task1_running_time = [Task1_running_time,time];  
    end
end

%% =================== Save Results ======================
title1 = {'Task1_acc_tar'};
results_acc = [task1_acc_tar'];
A = [title1;num2cell(results_acc)];

title2 = {'Task1_Best_C','Task1_Best_G'};
results_parameters = [Best_C_task1',Best_G_task1'];
B = [title2;num2cell(results_parameters)];

title3 = {'Task1_running_time'};
results_running_time = [Task1_running_time'];
C = [title3;num2cell(results_running_time)];

D = task1_src_predict_label;
E = task1_tar_predict_label;
