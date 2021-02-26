clear all;
clc;
load carsmall
x1 = Weight;
y = Displacement;
% creating our X matrix to use normal equation to find weights
temp = size(x1);
z = ones(temp(1), 1);
X = [z x1];
% our weights w0, w1 and w2 can be calculated as (X'X)^-1Xy
w = pinv(X.' * X) * (X.') * y;
sigma = sqrt((1/temp(1))*(y.'*y - y.'*X*w));
predy = zeros(temp(1), 1);
for i=1:temp(1)
   predy(i) = X(i, :)*w + normrnd(0, sigma) ;
end
scatter(x1,y)
hold on
scatter(x1,predy)
xlabel('weight')
ylabel('displacement')
title('MLE Relation Between weight and displacement')
grid on
legend('Data','MLE prediction', 'Location','best');
meansqaureerror = (1/temp(1))*(sum((predy-y).^2));
fprintf('mean square error is: %d \n', meansqaureerror);