function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Y = zeros(m,num_labels);
X = [ones(m, 1) X];
z2 = X*(Theta1');
a2 = sigmoid(X*(Theta1'));
a2 = [ones(m, 1) a2];
H = sigmoid(a2*(Theta2'));   %'
y1=y;
for i = 1:m,
  temp = y1(i);
  Y(i,temp) = 1;
end
Jtemp = (log(H).*Y + log(1-H).*(1-Y));
Jtemp2 = -(sum(sum(Jtemp))/m);
[a b] = size(Theta1);
[c d] = size(Theta2);
th1 = Theta1(:,2:b).^2;
th2 = Theta2(:,2:d).^2;
sumth1 = sum(sum(th1));
sumth2 = sum(sum(th2));
sumth = (sumth1 + sumth2)*(lambda/(2*m));
J = Jtemp2 + sumth;

for i=1:m,
  a1 = X(i:i,:)';                        %'
  ytemp = Y(i:i,:)';
  htemp = H(i:i,:)';
  D3 = htemp - ytemp;
  a2temp = a2(i:i,:)';                    %'
  z2temp = z2(i:i,:)';
  z2temp = [1 ; z2temp];
  D2 = (Theta2' * D3) .* sigmoidGradient(z2temp);
  D2 = D2(2:end);
  Theta2_grad = Theta2_grad + D3*a2temp';
  Theta1_grad = Theta1_grad + D2*a1';
end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;


Theta2_grad(:,2:d) = Theta2_grad(:,2:d) + (lambda/m).* Theta2(:,2:d);
Theta1_grad(:,2:b) = Theta1_grad(:,2:b) + (lambda/m).* Theta1(:,2:b);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
