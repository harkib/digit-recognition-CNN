function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));
input_od = zeros(size(input.data));

batch_size = size(input,2);
batch_len = size(input,1);


for b = 1:batch_size

    h_im1 = input.data(:,b);
    param_grad.b = transpose(output.diff(:,b)) + param_grad.b;
    param_grad.w = h_im1*transpose(output.diff(:,b)) + param_grad.w;
    input_od(:,b) = param.w*output.diff(:,b);

end

end
