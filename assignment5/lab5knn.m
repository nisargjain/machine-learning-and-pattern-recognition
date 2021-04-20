%NISARG JAIN
%17UCC039

clear all; clc;

% reading, cleaning and formating data

x = readtable('diabetes.csv','Format','auto');
X = x{:,:};
Y = X(:,9);
X = X(:, 1:8);

%creating training and test sets with 80/20 split
N = size(X,1);
idx = randperm(N);
split = 0.80;
x_train = X(idx(1:round(N*split)),:);
x_test = X(idx(round(N*split)+1:end),:);
y_train = Y(idx(1:round(N*split)),:);
y_test = Y(idx(round(N*split)+1:end),:);


%calculating distance matrix for every test data
numoftestdata = size(x_test,1);
numoftrainingdata = size(x_train,1);

%nearest neigbours
k= 7;

predictions = zeros(numoftestdata, 1);

for element = 1:numoftestdata
    testsample = x_test(element,:);
    distances = zeros(numoftrainingdata, 1);
       
    for i  = 1 : numoftrainingdata
        %calculating euclidean distance of testsample 
        %with each training data
        distances(i,1) =  sqrt(sum((testsample - x_train(i, :)).^2));
    end
       
    %finding k nearest neighbours
    [sorted , positions] = sort(distances, 'ascend');
    knearestneighbors = positions(1:k);
    knearestdistances = sorted(1:k);

    % Step 3 : Voting for maximum class
    for i=1:k
        A(i) = y_train(knearestneighbors(i),1);  
    end
    
    pred = mode(A);
    
    if pred == 1
        predictions(element) = 1;
    end
    
end

%we have our predictions in predictions array and test label in y_test
%lets find accuracy of our model

accuracy = round(sum(y_test == predictions)/numoftestdata, 3);

fprintf('accuracy of the KNN Model when k=%d is: %f \n',k, accuracy);

