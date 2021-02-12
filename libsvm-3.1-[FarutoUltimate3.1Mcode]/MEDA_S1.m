clc; clear all; close all;
%% =============================== 加载数据 =================================
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data1_pvc_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data8_pvc_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data10_pvc_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data1_ldpe_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data8_ldpe_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data10_ldpe_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data1_ps_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data8_ps_11Gra.csv')
load('D:\Document\Experiment\microplastics\SOIL\所有228NIR数据整合\data10_ps_11Gra.csv')

%% ============= 将每个数据集均划为两类 <high_大于1%> <low_小于1%> =============
pvc1_high = data1_pvc_11Gra(1:228,1:75);    pvc1_low = data1_pvc_11Gra(1:228,91:165);
pvc8_high = data8_pvc_11Gra(1:228,1:75);    pvc8_low = data8_pvc_11Gra(1:228,91:165);
pvc10_high = data10_pvc_11Gra(1:228,1:75);  pvc10_low = data10_pvc_11Gra(1:228,91:165);

ldpe1_high = data1_ldpe_11Gra(1:228,1:75);   ldpe1_low = data1_ldpe_11Gra(1:228,91:165);
ldpe8_high = data8_ldpe_11Gra(1:228,1:75);   ldpe8_low = data8_ldpe_11Gra(1:228,91:165);
ldpe10_high = data10_ldpe_11Gra(1:228,1:75); ldpe10_low = data10_ldpe_11Gra(1:228,91:165);

ps1_high = data1_ps_11Gra(1:228,1:75);   ps1_low = data1_ps_11Gra(1:228,91:165);
ps8_high = data8_ps_11Gra(1:228,1:75);   ps8_low = data8_ps_11Gra(1:228,91:165);
ps10_high = data10_ps_11Gra(1:228,1:75); ps10_low = data10_ps_11Gra(1:228,91:165);

pvc_lbl_high = zeros(1,75);   pvc_lbl_low = zeros(1,75);
ldpe_lbl_high = zeros(1,75);   ldpe_lbl_low = zeros(1,75);
ps_lbl_high = zeros(1,75);   ps_lbl_low = zeros(1,75);
for i = 1:75
    pvc_lbl_high(1,i) = 1;
    pvc_lbl_low(1,i) = 2;
    ldpe_lbl_high(1,i) = 3;
    ldpe_lbl_low(1,i) = 4;
    ps_lbl_high(1,i) = 5;
    ps_lbl_low(1,i) = 6;
end

y = cat(2,pvc_lbl_high,pvc_lbl_low,ldpe_lbl_high,ldpe_lbl_low,ps_lbl_high,ps_lbl_low);
y =y';

% 定义data_s1为加入PVC,LDPE,PS的soil1
% 定义data_s8为加入PVC,LDPE,PS的soil8
% 定义data_s10为加入PVC,LDPE,PS的soil10

data_s1 = cat(2,pvc1_high,pvc8_high,pvc10_high,pvc1_low,pvc8_low,pvc10_low);
data_s8 = cat(2,ldpe1_high,ldpe8_high,ldpe10_high,ldpe1_low,ldpe8_low,ldpe10_low);
data_s10 = cat(2,ps1_high,ps8_high,ps10_high,ps1_low,ps8_low,ps10_low);

data_s1 = data_s1';
data_s8 = data_s8';
data_s10 = data_s10';
% %% ================== 划分数据集 ====================
% X = data_s1';     y = y';    ratio = 0.7;
% [X_train, y_train, X_test, y_test] = split_train_test(X, y, ratio);
% train_sd1 = X_train; trainlbl = y_train;
% test_sd1 = X_test;   testlbl = y_test;
% 
% X = data_s8';        
% [X_train, y_train, X_test, y_test] = split_train_test(X, y, ratio);
% train_sd2 = X_train; trainlbl = y_train;
% test_sd2 = X_test;   testlbl = y_test;
% 
% X = data_s10';    
% [X_train, y_train, X_test, y_test] = split_train_test(X, y, ratio);
% train_sd3 = X_train; trainlbl = y_train;
% test_sd3 = X_test;   testlbl = y_test;

%% ============== Organize Domain Datasets ================
save('D:\data\Datasets\soil_s1_s8_s10\data_s1.mat','data_s1');
save('D:\data\Datasets\soil_s1_s8_s10\data_s8.mat','data_s8');
save('D:\data\Datasets\soil_s1_s8_s10\data_s10.mat','data_s10');
save('D:\data\Datasets\soil_s1_s8_s10\labels.mat','y');

%% ======================= MEDA ===========================
labels = cell2mat(struct2cell(load('D:\data\Datasets\soil_s1_s8_s10\labels.mat')));
str_domains = {'data_s1', 'data_s8', 'data_s10'};
list_acc = [];
for i = 1 : 3
    for j = 1 : 3
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        fts = cell2mat(struct2cell(load(['D:\data\Datasets\soil_s1_s8_s10/' src '.mat'])));     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = zscore(fts,1);    clear fts
        Ys = labels;           
        
        fts = cell2mat(struct2cell(load(['D:\data\Datasets\soil_s1_s8_s10/' tgt '.mat'])));     % target domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = zscore(fts,1);     clear fts
        Yt = labels;            
        
        % meda
        options.d = 50;
        options.rho = 1.0;
        options.p = 10;
        options.lambda = 10.0;
        options.eta = 0.1;
        options.T = 30;
        [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
    end
end