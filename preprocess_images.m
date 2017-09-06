% Script for reading, resizing and calculating the crops for the core
% biopsy samples. This must be run first in order to generate data for the
% CNNs.

clear all;
close all;

data_path = '../data_original';
sub_path = {'Glas1', 'Glas2', 'Glas3', 'Glas4', 'Glas5', ...
    'Glas6', 'Glas7', 'Glas8', 'Glas9'};
colorpack = 'ColorPack 1'; % If using the different color packs
final_paths = {};
save_path = '../data_1';


% Get paths
for glas = 1:length(sub_path)
    folders = dir(fullfile(data_path, sub_path{glas}));
    image_folders = {};

    c = 1;
    for image_nr = 1:length(folders)
        name = folders(image_nr).name;
        if length(name) <= 2 || length(findstr(name, 'Ctrl')) > 0
           continue;
        end

        image_folders{c} = name;
        c = c + 1;
    end

    image_paths = fullfile(data_path, sub_path{glas}, image_folders); 

    for image_nr = 1:length(image_paths)
        files = dir(image_paths{image_nr});
        
        images = {files(:).name};
        
        
        
        for file = 1:length(files)
            % Change colorpack to '.tiff' and split{4} to split{2} when
            % returning and running on the single color pack.
            if ~isempty(strfind(files(file).name, '.tiff'))
               final_paths{size(final_paths, 1)+1, 1} = ...
                   fullfile(image_paths{image_nr}, files(file).name);
               final_paths{size(final_paths, 1), 2} = sub_path{glas};
               split = strsplit(image_paths{image_nr});
               final_paths{size(final_paths, 1), 3} = split{2};
            end
        end
    end
end

% Crop images by iteration
output_size = 256
scale = 0.3
limit_cancer = 10;
limit_include = 0.01;
color = reshape([206 162 98], 1, 1, 3);

for k = 1:size(final_paths, 1)
    
    img = imread(final_paths{k, 1});
    img  = imresize(img, scale);
    
    [m, n, ~] = size(img);
    
    if (m - (output_size - 1)) < 0 || (n - (output_size - 1)) < 0
        m
        n
        final_paths{k, 1}
        continue;
    end
    
    img = imresize(img, [m - mod(m, output_size/2), ...
        n - mod(n, output_size/2)]);
    
    [m, n, ~] = size(img);
    
    img_done = 0;
    for i = output_size:(output_size/2):m
        for j = output_size:(output_size/2):n
            img_crop = img((i-(output_size-1)):i, (j-(output_size-1)):j,:);
            
            img_mid = img_crop(64:192, 64:192, :);
            
            img_logical = sum(abs(double(img_mid) - color), 3) < ...
                limit_cancer;
            foreground = sum(sum(img_logical)) / (128*128);          
            
            if foreground > limit_include
                img_done = img_done + 1;
                
                name = fullfile(save_path, [final_paths{k, 2}, '_', ... 
                    final_paths{k, 3}, '_', num2str(img_done), '.tiff']);
                imwrite(img_crop, name);
            end
            
        end 
    end
    
    disp(sprintf('Cropped image: %d/%d \t crops: %d', ...
        [k, size(final_paths, 1), img_done]));
end

%{
% Crop images by randomcrop
scale = 0.15;
img_tot = 20;
itr_limit = 500;
limit_cancer = 10;
limit_include = 0.001;
color = reshape([206 162 98], 1, 1, 3);
output_size = [256, 256];
for k = 1:size(final_paths, 1)
    disp(sprintf('Cropping image: %d/%d', [k, size(final_paths, 1)]));
    
    img = imread(final_paths{k, 1});
    img  = imresize(img, scale);
    
    [m, n, ~] = size(img);
    
    if (m - output_size(1) - 1) < 0 || (n - output_size(2) - 1) < 0
        o1 = (m - output_size(1) - 1)
        o2 = (n - output_size(2) - 1)
        final_paths{k, 1}
        continue;
    end
        
    
    img_done = 1;
    itr = 0;
    while img_done <= img_tot
        r1 = randi(m - output_size(1) - 1);
        r2 = randi(n - output_size(2) - 1);
    
        img_crop = img(r1:(r1 + output_size(1) - 1), ...
            (r2:r2 + output_size(2) - 1), :);
        
        img_logical = sum(abs(double(img_crop) - color), 3) < limit_cancer;
        foreground = sum(sum(img_logical)) / ...
            (output_size(1)*output_size(2));
        
        if foreground > limit_include
            name = fullfile(save_path, [final_paths{k, 2}, '_', ... 
                final_paths{k, 3}, '_', num2str(img_done), '.tiff']);
            imwrite(img_crop, name);
            
            img_done = img_done + 1;
        end
        
        itr = itr + 1;
        if itr > itr_limit
            disp('Iteration limit reached');
            break;
        end
    end
    
end
%}


%{
% Resize images
for k = 1:size(final_paths, 1)
    disp(sprintf('Resizing image %d/%d', [k,size(final_paths, 1)]));
    img = imread(final_paths{k, 1});
    img  = imresize(img, [227, 227]);
    name = fullfile(save_path, [final_paths{k, 2}, '_', ...
        final_paths{k, 3}, '.tiff']);
    imwrite(img, name);
end
%}

%{
% Calculate size and plot sizes of all images
outliers = [];
upper_lim = 8000;
lower_lim = 6000;
for k = 1:size(final_paths, 1)
    disp(sprintf('Reading size %d/%d', [k,size(final_paths, 1)]));
    img = imread(final_paths{k, 1});
    [m, n, ~] = size(img);
    sizes = [sizes, m+n];
end
plot(sizes)
%}
