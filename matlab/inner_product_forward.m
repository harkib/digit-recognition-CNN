function [output] = inner_product_forward(input, layer, param)

% given:  W is a two dimensional matrix of m × n size where
% n is the dimensionality of the previous layer and m is the
% number of neurons in this layer, w must be transposed
param.w = transpose(param.w);
param.b = transpose(param.b);



%d = size(input.data, 1);
%k = size(input.data, 2); % batch size
%n = size(param.w, 2);



% Replace the following line with your implementation.
% output.data = zeros([n, k]);

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
h_out = size(param.w, 1);
w_out = 1;

output.height = h_out;
output.width = w_out;
output.channel = 1;
output.batch_size = batch_size;
output.data = zeros([h_out*w_out,batch_size]);

for b = 1:batch_size
    col_in = input.data(:,b);
    col_out = param.w*col_in + param.b;
    output.data(:,b)=col_out;
end

end

















