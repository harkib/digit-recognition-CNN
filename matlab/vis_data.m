close all;


layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
%imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);

% Fill in your code here to plot the features.

% Cov without normalization
layer_out = output{2};
height = layer_out.height;
width = layer_out.width;
channel = layer_out.channel;
batch_size = layer_out.batch_size;
data = layer_out.data;
feature_maps = reshape(data,height,width,channel);

figure;
for i = 1:channel
    subplot(4,5,i);
    imshow(feature_maps(:,:,i));
end
sgtitle('Conv Layer No Normalization');
savefig ('results\vis\ConvNoNorm.fig')


% Cov with normalization
layer_out = output{2};
height = layer_out.height;
width = layer_out.width;
channel = layer_out.channel;
batch_size = layer_out.batch_size;
data = layer_out.data;
data = normalize(data,'range');
feature_maps = reshape(data,height,width,channel);

figure;
for i = 1:channel
    subplot(4,5,i);
    imshow(feature_maps(:,:,i));
end
sgtitle('Conv Layer with Normalization');
savefig ('results\vis\ConvNorm.fig')


% Relu
layer_out = output{3};
height = layer_out.height;
width = layer_out.width;
channel = layer_out.channel;
batch_size = layer_out.batch_size;
data = layer_out.data;
feature_maps = reshape(data,height,width,channel);

figure;
for i = 1:channel
    subplot(4,5,i);
    imshow(feature_maps(:,:,i));
end
sgtitle('Relu Layer');
savefig ('results\vis\Relu.fig')

