% Nisarg Jain
% 17ucc039

clc; clear all;

%importing, formating and cleaning data
X = readtable('olympic.csv', 'Format', 'auto');

%lets select 100m data from the dataset and we
%we focus on men's running time only for the 
%purpose of this lab
Y = X(:,8);
X = X(:, 2:4);
X.Event = categorical(X.Event);
Y = Y(X.Event == '100M Men', :);
X = X(X.Event == '100M Men', :);
X = X(:, 3);
m = size(Y, 1);
X = X{:,:};
ytemp = Y;
y = zeros(m,1);

%-------------------------------------------
%changing target values to meaningful numbers

for i = 1:m
    ytemp(i,1) = convertCharsToStrings(ytemp(i,1));
end

ytemp = ytemp{:,:};

for i = 1:m
    if strcmp(ytemp(i, 1),'None')
        y(i,1) = NaN;
    else 
        y(i,1) = str2double(ytemp(i,1));
    end
end


%filling missing values
meanofy = mean(y, 'omitnan');
for i = 1:m
    if isnan(y(i))
        y(i) = meanofy;
    end
end


%plot data
figure;
plot(X, y, 'bo','markersize',10);

%creating X
sizeofx = size(X, 1);
temp = ones(sizeofx, 1);
x = cat(2, temp, X);
X = X.^2;
x = cat(2, x, X);

%kth order polynomial: we assume k = 2, meaning quadratic model
% assuming our prior as gaussian, initial parameters are:
mu0 = [0 0 0].';
sigma0 = [100 0 0 ; 0 1 0; 0 0 1];
ss = 0.5;

%with given prior we know our posterior as:
sigmapos = pinv((x.'*x)/ss^2 + inv(sigma0));
mupos = sigmapos*((x.'*y)/ss^2 + inv(sigma0)*mu0);

%making prediction for year 2020
year = 2020;
%scaling according to previous scheme

xvec = [1 year year^2].';

%model posterior given year
munew = xvec.'*mupos;
sigmanew = ss^2 + xvec.'*sigmapos*xvec;

%plotting for year 2020
figure;
j = 7:.01:12;
k = normpdf(j,munew, sigmanew);
plot(j,k);
xlabel('running time distribution for year 2020');
hold off;
