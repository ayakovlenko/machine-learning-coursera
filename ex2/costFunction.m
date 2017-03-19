function [J, gradient] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

yPred = sigmoid(X * theta);
J = mean(-y .* log(yPred) - (1 - y) .* log(1 - yPred));

Err = yPred - y;
gradient = mean(Err .* X)';

end
