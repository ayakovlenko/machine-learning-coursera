function [J, gradient] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);

yPred = sigmoid(X * theta);
J = mean(-y .* log(yPred) - (1 - y) .* log(1 - yPred)) ...
    + (lambda / (2 * m)) * (theta(2:end)' * theta(2:end));

Err = yPred - y;

gradient0 = mean(Err .* X(:, 1));
gradient1 = mean(Err .* X(:, 2:end))' + (lambda / m) * theta(2:end);

gradient = [gradient0; gradient1];

end
