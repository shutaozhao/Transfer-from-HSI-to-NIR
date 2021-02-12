%% ========================== Randomly divide the dataset =============================
function [X_train, y_train,  X_test, y_test] = split_train_test(X, y, ratio)
%  Input: X is the spectral feature matrix; y is the corresponding class
%  Input: ratio is the division percentage(trainning samples / all samples)
%  Output: train set - X_train & y_train; test set - X_test & y_test

m = size(X, 1);
y_labels = unique(y); 
k = length(y_labels);
d = [1:m]';

X_train = [];
y_train= [];

for i = 1:k
    comm_i = find(y == y_labels(i));
    if isempty(comm_i) 
        continue;
    end
    size_comm_i = length(comm_i);
    rp = randperm(size_comm_i); % random permutation
    rp_ratio = rp(1:floor(size_comm_i * ratio));
    ind = comm_i(rp_ratio);
    X_train = [X_train; X(ind, :)];
    y_train = [y_train; y(ind, :)];
    d = setdiff(d, ind);
end

X_test = X(d, :);
y_test = y(d, :);

end
