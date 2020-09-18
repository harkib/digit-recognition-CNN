function [input_od] = relu_backward(output, input, layer)

% Replace the following line with your implementation.
input_od = zeros(size(input.data));
batch_size = size(input,2);
batch_len = size(input,1);


for b = 1:batch_size
    for i = 1:batch_len
        h_im1 = input.data(i,b);
        if h_im1 > 0
            input_od(i,b) = output.diff(i,b);
        else 
            input_od(i,b) = 0;
        end
    end
end
end
