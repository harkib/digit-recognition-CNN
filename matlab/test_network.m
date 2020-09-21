%% Network defintion
clear all;
close all;

layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix
perdictions = zeros(size(ytest));
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    P_max = zeros([1,size(P,2)]);
    for j = 1:size(P,2)
        P_max(j) = find(P(:,j) == max(P(:,j))); 
    end
    perdictions(i:i+99)=P_max;
end     

%% Confusionmat
C = confusionmat(ytest,perdictions);
figure;
confusionchart(C);
title("Confusion Chart");
savefig('results\confusionchart.fig');