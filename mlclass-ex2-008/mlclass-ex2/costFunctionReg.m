function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

sig = sigmoid(X*theta);
h1 = 1.0 .* log(sig);
h2 = 1.0 .* log(1-sig);
y2 = 1 .- y;
temp = y .* h1 + y2 .* h2;
s = -sum(temp)

sqtheta = (sum(theta .^ 2) - theta(1)^2)* (lambda/(2*m));
J = s/m + sqtheta; 

v = []

for i = 1:size(X, 2)
		if i==1
			pred = (sigmoid(X*theta)-y) .* X(:, i);
			S=sum(pred)/m;
			v=[v, S];
		end
		if i>1
			pred = (sigmoid(X*theta)-y) .* X(:, i);
			S=sum(pred)/m + (lambda/m)*theta(i);
			v=[v, S];
		end	
end
v = v'
grad = v




% =============================================================

end
