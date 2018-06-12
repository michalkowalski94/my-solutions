function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
st = size(theta);
J = 1/(2*m)*sum((h-y).^2)+lambda/(2*m)*sum(theta(2:st).^2);

%WHEN YOU ARE CHANGING VARIABLES IN FRICKIN DATA - NOTIFY PEOPLE VIA FRICKIN EMAIL
%I'VE SPENT AN HOUR TRYING TO FIGURE OUT WHAT'S WRONG WITH CORRECTLY IMPLEMENTED CODE
%I'VE DOWNLOADED ONCE AGAIN DATASET AND HERE YOU ARE GOD DAMN IT - FIRST IMPLEMENTATION WAS GOOD IMPLEMENTATION
%ARGRGHHHH

grad = (1/m)*(X'*(h-y));
grad(2:st) = grad(2:st)+(lambda/m)*theta(2:st);






% =========================================================================

grad = grad(:);

end
