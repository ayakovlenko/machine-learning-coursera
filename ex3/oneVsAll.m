function [all_theta] = oneVsAll(X, y, K, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, K, lambda) trains K
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(K, n + 1);

X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', 50);

for k = 1:K
    y_k = y == k;

    costFn = @(t)(lrCostFunction(t, X, y_k, lambda));
    all_theta(k, :) = fminunc(costFn, all_theta(k, :)', options);
end

end
