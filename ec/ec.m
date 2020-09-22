%% load model 
layers = get_lenet();
load lenet.mat

%% load data, find digit, and perdict
close all;

src_im_names = ["image1.JPG","image2.JPG","image3.png","image4.jpg"];

for n = 1:size(src_im_names,2)
    
    %%remove backgorund (->black)
    % grey scale and turn digits white
    src_im = imread(src_im_names(n));
    src_im = double(rgb2gray(src_im));
    for i = 1:size(src_im,1)
        for j = 1:size(src_im,2)
            src_im(i,j) = 1 - src_im(i,j)/255;
        end
    end

    %https://www.mathworks.com/help/images/correcting-nonuniform-illumination.html
    se = strel('disk',20);
    background = imopen(src_im,se);
    src_im = src_im - background;
    src_im = imadjust(src_im);
    src_im = imbinarize(src_im);
    src_im = bwareaopen(src_im,50);
    
    figure;
    imshow(src_im);
    title(src_im_names(n) + " Background Removed");
    savefig("results\ec\background\"+num2str(n) + "_scr_backgroundRM.fig");
    
    %%isolate image
    conn_im = bwlabel(src_im);
    n_sub_ims = max(conn_im,[],'all');
    sub_ims = zeros([28,28,n_sub_ims]);    
    
    pad_row = size(src_im,2)/60 ;
    pad_col = pad_row*1.2;
    max_col = size(src_im, 2);
    max_row = size(src_im, 1);
    
    for s = 1:n_sub_ims
        [r, c] = find(conn_im==s);
        max_r = round(min(max(r)+pad_row,max_row));
        max_c = round(min(max(c)+pad_col,max_col));
        min_r = round(max(min(r)-pad_row,1));
        min_c = round(max(min(c)-pad_col,1));
    
        
        raw_sub_im = src_im(min_r:max_r,min_c:max_c); 
        sub_ims(:,:,s) = transpose(imresize(raw_sub_im,[28,28]));
    end
    
    %%batch
    x = reshape(sub_ims,28*28,n_sub_ims);
    layers{1}.batch_size = n_sub_ims;
    
    
    %%perdict
    [output, P] = convnet_forward(params, layers,x);
    perdictions = zeros([1,size(P,2)]);
    for j = 1:size(P,2)
        perdictions(j) = find(P(:,j) == max(P(:,j))) - 1; 
    end
    
    %%display results
    figure;
    root_n = ceil(sqrt(n_sub_ims));
    for s = 1:n_sub_ims
        subplot(root_n,root_n,s);
        imshow(transpose(reshape(x(:,s),28,28)));
        title(num2str(perdictions(s)));
    end
    sgtitle(src_im_names(n));
    savefig ("results\ec\perdictions\" + num2str(n) + ".fig");
    
end