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
predy = X*w;
scatter(x1,y)
hold on
plot(x1,predy)
xlabel('weight')
ylabel('displacement')
title('Linear Regression Relation Between weight and displacement')
grid on
legend('Data','Slope', 'Location','best');
meansqaureerror = (1/temp(1))*(sum((predy-y).^2));
fprintf('mean square error is: %d \n', meansqaureerror);
