%Nisarg Jain
%17ucc039
clc; clear all;

%importing, cleaning and processing the data
data = readtable('cancer.csv', 'Format', 'auto');
X = data(:, 3:32);
y = data(:, 2);
X = X{:,:};

%label encoding y
sizeofy = size(y,1);
ytemp = y;
y = zeros(sizeofy, 1);

for i = 1:sizeofy
    ytemp(i,1) = convertCharsToStrings(ytemp(i,1));
end

ytemp = ytemp{:,:};

for i = 1:sizeofy
    if strcmp(ytemp(i, 1),'M')
        y(i,1) = 2;
    else 
        y(i,1) = 1;
    end
end

%creating training and test sets with 80/20 split
N = size(X,1);
idx = randperm(N);
split = 0.80;
X_train = X(idx(1:round(N*split)),:);
X_test = X(idx(round(N*split)+1:end),:);
y_train = y(idx(1:round(N*split)),:);
y_test = y(idx(round(N*split)+1:end),:);

%No of output categories
input_layer_size  = size(X_train,2);  
hidden_layer_size = 30;   
num_labels = 2; 
m = size(X_train, 1);


%theta inititalization where theta is weight vectors
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = 0.5;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X_train, y_train, lambda);
               
options = optimset('MaxIter', 50);


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X_test);
predtrain = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(predtrain == y_train)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);


% -------------FUNCTIONS-------------------------------------------------- 

function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate functioN.
% Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    length = options.MaxIter;
else
    length = 100;
end


RHO = 0.01;                            % a bunch of constants for line searches
SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 100;                                      % maximum allowed slope ratio

argstr = ['feval(f, X'];                      % compose string used to call function
for i = 1:(nargin - 3)
  argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                      % get function value and gradient
i = i + (length<0);                                            % count epochs?!
s = -df1;                                        % search direction is steepest
d1 = -s'*s;                                                 % this is the slope
z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
  X = X + z1*s;                                             % begin line search
  [f2 df2] = eval(argstr);
  i = i + (length<0);                                          % count epochs?!
  d2 = df2'*s;
  f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
  if length>0, M = MAX; else M = min(MAX, -length-i); end
  success = 0; limit = -1;                     % initialize quanteties
  while 1
    while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
      limit = z1;                                         % tighten the bracket
      if f2 > f1
        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
      else
        A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
      end
      if isnan(z2) || isinf(z2)
        z2 = z3/2;                  % if we had a numerical problem then bisect
      end
      z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
      z1 = z1 + z2;                                           % update the step
      X = X + z2*s;
      [f2 df2] = eval(argstr);
      M = M - 1; i = i + (length<0);                           % count epochs?!
      d2 = df2'*s;
      z3 = z3-z2;                    % z3 is now relative to the location of z2
    end
    if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
      break;                                                % this is a failure
    elseif d2 > SIG*d1
      success = 1; break;                                             % success
    elseif M == 0
      break;                                                          % failure
    end
    A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
    B = 3*(f3-f2)-z3*(d3+2*d2);
    z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
    if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 % num prob or wrong sign?
      if limit < -0.5                               % if we have no upper limit
        z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
      else
        z2 = (limit-z1)/2;                                   % otherwise bisect
      end
    elseif (limit > -0.5) && (z2+z1 > limit)         % extraplation beyond max?
      z2 = (limit-z1)/2;                                               % bisect
    elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
      z2 = z1*(EXT-1.0);                           % set to extrapolation limit
    elseif z2 < -z3*INT
      z2 = -z3*INT;
    elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))  % too close to limit?
      z2 = (limit-z1)*(1.0-INT);
    end
    f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
    z1 = z1 + z2; X = X + z2*s;                      % update current estimates
    [f2 df2] = eval(argstr);
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d2 = df2'*s;
  end                                                      % end of line search

  if success                                         % if line search succeeded
    f1 = f2; fX = [fX' f1]';
    fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
    s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    d2 = df1'*s;
    if d2 > 0                                      % new slope must be negative
      s = -df1;                              % otherwise use steepest direction
      d2 = -s'*s;    
    end
    z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
    d1 = d2;
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
    if ls_failed || i > abs(length)          % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    s = -df1;                                                    % try steepest
    d1 = -s'*s;
    z1 = 1/(1-d1);                     
    ls_failed = 1;                                    % this line search failed
  end
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
end
fprintf('\n');
end


function [J grad] = nnCostFunction(nn_params,input_layer_size, ...
                        hidden_layer_size,num_labels, X, y, lambda)
%Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, 
% the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);


%PART 1. Forward propogation and calculation of costfunction
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

K = num_labels;
X = [ones(m,1) X];

for i = 1:m
	X_i = X(i,:);
	h_of_Xi = sigmoid( [1 sigmoid(X_i * Theta1')] * Theta2' );
	y_i = zeros(1,K);
	y_i(y(i)) = 1;
	J = J + sum( y_i .* log(h_of_Xi) + (1 - y_i) .* log(1 - h_of_Xi));
end

J = (-1 / m) * J;
Theta1s=Theta1(:,2:end);
Theta2s=Theta2(:,2:end);

% Adding regularization term
J = J + (lambda / (2 * m) * (sum(sum(Theta1s.^2)) + sum(sum(Theta2s.^2)))); 

%PART 2: Implement the backpropagation algorithm to compute the gradients

delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));

for t = 1:m
	a1 = X(t,:);  
	z2 = a1 * Theta1';
	a2 = [1 sigmoid(z2)];
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);
	yi = zeros(1,K);
	yi(y(t)) = 1;
	
	delta_3 = a3 - yi;
	delta_2 = delta_3 * Theta2 .* sigmoidGradient([1 z2]);
	
	delta_accum_1 = delta_accum_1 + delta_2(2:end)' * a1;
	delta_accum_2 = delta_accum_2 + delta_3' * a2;
end

Theta1_grad = delta_accum_1 / m;
Theta2_grad = delta_accum_2 / m;


% Part 3: Implement regularization with the cost function and gradients.

Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) ...
                            + lambda / m * Theta1(:, 2:input_layer_size+1);
Theta2_grad(:, 2:hidden_layer_size+1)= Theta2_grad(:, 2:hidden_layer_size+1) ...
                        + lambda / m * Theta2(:, 2:hidden_layer_size+1);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end


function p = predict(Theta1, Theta2, X)
    %to predict we just need theta 1 and theta 2 and 
    %forward propogate
   
    m = size(X, 1);
    num_labels = size(Theta2, 1);
    p = zeros(size(X, 1), 1);
    h1 = sigmoid([ones(m, 1) X] * Theta1');
    h2 = sigmoid([ones(m, 1) h1] * Theta2');
    [dummy, p] = max(h2, [], 2);
    
end


function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end


function g = sigmoidGradient(z)
    g = zeros(size(z));
    g=sigmoid(z).*(1-sigmoid(z));
end


function W = randInitializeWeights(L_in, L_out)

%Randomly initialize the weights of a layer 
%with L_in incoming connections and L_out outgoing connections

    W = zeros(L_out, 1 + L_in);
    EPSILON=sqrt(6)./(sqrt(L_in)+sqrt(L_out+1));
    W = rand(L_out, 1 + L_in) * 2 * EPSILON-EPSILON;
end


