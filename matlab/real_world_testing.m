clear all;
close all;


%% load model
layers = get_lenet();
load lenet.mat

%% load data
ims = ["images\web1.jpg","images\web2.jpg","images\web3.jpg","images\web4.jpg","images\web5.jpg"];
layers{1}.batch_size = size(ims,2);
x = zeros([28*28,size(ims,2)]);
for i = 1:size(ims,2)
    col = prep_im(imread(ims(i)));
    x(:,i) = col(:,1);
end

%% perdict
[output, P] = convnet_forward(params, layers,x);
perdictions = zeros([1,size(P,2)]);
for j = 1:size(P,2)
    perdictions(j) = find(P(:,j) == max(P(:,j))) - 1; 
end

answers = [1,5,2,8,5];
disp("Answers:");
disp(answers);
disp("Perdictions:");
disp(perdictions);



function [y] = prep_im(x)
    x = imresize(x, [28,28]);
    x = rgb2gray(x);
    x = transpose(x);
    x = reshape(x,28*28,1);
    
    y = zeros(size(x));
    for i = 1:size(x,1)
       y(i) = 1 - x(i)/255; 
    end
    
end