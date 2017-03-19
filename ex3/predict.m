function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

addBias = @(M)([ones(size(M, 1), 1), M]);

Thetas = {Theta1, Theta2};

A = X;
for Theta = Thetas
    A = sigmoid(addBias(A) * cell2mat(Theta)');
end

[~, p] = max(A, [], 2);

end
