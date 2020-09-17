function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
%input_n.height = h_in;
%input_n.width = w_in;
%input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 

output.height = h_out;
output.width = w_out;
output.channel = num;
output.batch_size = batch_size;
output.data = zeros([h_out*w_out*num,batch_size]);

for b = 1:batch_size
    
    feature_map = zeros([h_out,w_out,num]);

    % get single sample as matrix
    col_in = input.data(:,b);
    mat_in = reshape(col_in,h_in,w_in,c);

    %pad matrix
    if pad ~= 0
        mat_in = padarray(mat_in,[pad,pad,0],0,'both');
    end 
    
    % itterate through k by k sections and build feature map
    for i = 1:stride:size(mat_in,1)-k + 1
        for j = 1:stride:size(mat_in,2)-k + 1
            mat_sub = mat_in(i:i+k-1,j:j+k-1,:);
            row_sub = reshape(mat_sub,1,k*k*c);
            row_out_sub = row_sub*param.w + param.b;
            feature_map(round(i/stride),round(j/stride),:) = row_out_sub(1,:);
        end
    
    % colapse feature map into col of data
    output.data(:,b)=reshape(feature_map,h_out*w_out*num,1);
    end
    
end

end

