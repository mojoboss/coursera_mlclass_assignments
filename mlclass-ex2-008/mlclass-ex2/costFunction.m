function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sig = sigmoid(X*theta);
h1 = 1.0 .* log(sig);
h2 = 1.0 .* log(1-sig);
y2 = 1 .- y;
temp = y .* h1 + y2 .* h2;
s = -sum(temp)
J = s/m; 

v = []

for i = 1:size(X, 2)
		pred = (sigmoid(X*theta)-y) .* X(:, i);
		S=sum(pred);
		v=[v, S];
end
v = v'
v = (1/m) * v
grad = v



% =============================================================

end
