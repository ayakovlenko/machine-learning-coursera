function [J, grad] = lrCostFunction(theta, X, y, lambda)
% LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
% regularization
%    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%    theta as the parameter for regularized logistic regression and the
%    gradient of the cost w.r.t. to the parameters.

m = length(y);

theta1 = theta(2:end);

yPred = sigmoid(X * theta);
J = mean(-y .* log(yPred) - (1 - y) .* log(1 - yPred)) ...
    + (lambda / (2 * m)) * (theta1' * theta1);

Err = yPred - y;
grad0 = mean(Err .* X(:, 1));
grad1 = mean(Err .* X(:, 2:end))' + (lambda / m) * theta1;

grad = [grad0; grad1];

end
