function [output] = relu_forward(input)
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

% Replace the following line with your implementation.

output.data = zeros(size(input.data));
for i = 1:size(input.data,1)
    for j = 1:size(input.data,2)
        output.data(i,j) = max(input.data(i,j),0);
    end
end

end
