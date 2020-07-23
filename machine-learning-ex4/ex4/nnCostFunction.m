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

%Part-1-a
a1 = [ones(m,1) X];   %Add a column of 1's to X (the first column), and it becomes 'a1'.
z2 = a1*Theta1';     %Multiply by Theta1 and you have 'z2'
a2= sigmoid(z2);     %Compute the sigmoid() of 'z2', then add a column of 1's, and it becomes 'a2'
a2 = [ones(m,1) a2]; %add a subscript 0 superscript 2
z3 = a2*Theta2';     %Multiply by Theta2
a3 = sigmoid(z3);
h = a3;
%For the confusion that is created in below operations check the resources for week5 and also do visit the discussion threads given in these resources.
y_matrix = eye(num_labels)(y,:);
term1 = trace(y_matrix'*log(h));     %You can also use Double sum here-> term1 = sum(sum(y_matrix.*log(h)))
term2 = trace((1-y_matrix)'*log(1-h));   %here also -> term2 = sum(sum((1-y_matrix).*log(1-h)))
J = (-1/m)*(term1+term2);

%Part 1-b
Regularization = sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2));  %Remember its better to use double sum here instead of trace as it's complicated
J = J + (lambda/(2*m))*Regularization;

%Part-2

for t = 1:m

	% For the input layer, where layer=1:
	a1 = [1; X(t,:)'];  %401*1

	% For the hidden layers, where layer=2:
	z2 = Theta1 * a1;   % (25*401)*(401*1)   MULTIPLICATION ORDER IS SET ACCORDINGLY TO AVOID DIMENSION PROBLEMS 
	a2 = [1; sigmoid(z2)];  %(26*1)

	z3 = Theta2 * a2;   % (10*26)*(26*1)     MULTIPLICATION ORDER IS SET ACCORDINGLY TO AVOID DIMENSION PROBLEMS
	a3 = sigmoid(z3);    %(10*1)

	yy = ([1:num_labels]==y(t))';  %TRANSPOSING TO GET A COLUMN VECTOR
	% For the delta values:
	delta_3 = a3 - yy;   % (10*1)

	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];   %  ((26*10)*(10*1))=(26*1) ADDING BIAS COLUMN AS WE DONT SIGMOID BIAS UNIT
	delta_2 = delta_2(2:end);   % skipping sigma2(0) (25*1) Taking ofF the bias row

	% delta_1 is not calculated because we do not associate error with the input    

	% Big delta update
	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
end

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

%Part-3

% Regularization


% 
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
% 

% 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
