function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    % output.data = zeros([h_out, w_out, c, batch_size]);
    for b = 1:batch_size

        pooled = zeros([h_out,w_out,c]);
        
        % get single sample as matrix
        col_in = input.data(:,b);
        mat_in = reshape(col_in,h_in,w_in,c);
        
        %pad matrix
        if pad ~= 0
            mat_in = padarray(mat_in,[pad,pad,0],0,'both');
        end
        % itterate through k by k sections and build pooled output
        for i = 1:stride:size(mat_in,1)-k + 1
            for j = 1:stride:size(mat_in,2)-k + 1
                for w = 1:c
                    mat_sub = mat_in(i:i+k-1,j:j+k-1,w);
                    row_sub = reshape(mat_sub,1,k*k);
                    max_sub = max(row_sub);
                    pooled(round(i/stride),round(j/stride),w) = max_sub;
                end
            end

        % colapse feature map into col of data
        output.data(:,b)=reshape(pooled,h_out*w_out*c,1);
        end
    end
    
end

