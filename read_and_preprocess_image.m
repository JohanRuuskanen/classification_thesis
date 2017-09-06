function [img] = read_and_preprocess_image(filename, channel_mean)
    % Function for reading, normalizing and data augmentation for the CNN 
    % function.

    % Reading input image and transform to double
    img = double(imread(filename));
    
    % Color stack if gray
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
  
    % Resizing
    img = imresize(img, [256, 256]);

    % Data augmentation
    r1 = randi(4) - 1;
    r2 = randi(2) - 1;
    r3 = randi(256 - 227 + 1, 2, 1);
    
    % Rotation
    img = imrotate(img, r1*90);
    
    % Mirroring
    if r2 == 1
        img = fliplr(img);
    end
    
    % Cropping
    img = img(r3(1):(227 + r3(1) - 1), r3(2):(227 + r3(2) - 1), :);
    
    % Remove mean and std
    img = (img - channel_mean);
end

