function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    Err = X * theta - y;
    update = alpha * mean(Err .* X)';
    theta = theta - update;
    J_history(iter) = computeCost(X, y, theta);
end

end
