% Run for cropping on the images containing no cancer markers. 

clear all;
close all;

% Define parameters and initiate variables
output_size = 256
scale = 0.3
limit_cancer = 10;
limit_include = 0.01;
color = reshape([206 162 98], 1, 1, 3);

data_path_ref = '../data_original';
data_path_crop = '../data_original/FärgPaket Export 2 No Tumor';
sub_path = {'Glas1', 'Glas2', 'Glas3', 'Glas4', 'Glas5', ...
    'Glas6', 'Glas7', 'Glas8', 'Glas9'};
save_path = '../data_noTumor';

position_map = containers.Map;
final_paths_ref = get_paths(data_path_ref, sub_path);
final_paths_crop = get_paths(data_path_crop, sub_path);

% Extract the image crop positions from data_path_ref
disp('Extracting crop positions');
for k = 1:size(final_paths_ref, 1)
    
    img = imread(final_paths_ref{k, 1});
    img  = imresize(img, scale);
    
    [m, n, ~] = size(img);
    
    if (m - (output_size - 1)) < 0 || (n - (output_size - 1)) < 0
        m
        n
        final_paths_ref{k, 1}
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
                
                key = [final_paths_ref{k, 2}, '_', final_paths_ref{k, 3}];
                
                if position_map.isKey(key)
                    pos = position_map(key);
                else
                    pos = [];
                end
                
                pos = [pos; i, j];
                
                position_map(key) = pos;
                
            end
            
        end 
    end
    disp(sprintf('Extracted crop pos: %d/%d \t crops: %d', ...
         [k, size(final_paths_ref, 1), img_done]));
end

% Crop the images with no tumor tissue from the reference
fprintf('\n');
disp('Cropping from extracted positions');
for k = 1:size(final_paths_crop, 1)
    
    img = imread(final_paths_crop{k, 1});
    
    % Multiply scale by 2 since the image package is 2 times smaller than
    % the reference
    img  = imresize(img, scale*2); 
    
    [m, n, ~] = size(img);
    
    if (m - (output_size - 1)) < 0 || (n - (output_size - 1)) < 0
        m
        n
        final_paths_crop{k, 1}
        continue;
    end
    
    img = imresize(img, [m - mod(m, output_size/2), ...
        n - mod(n, output_size/2)]);
    
    [m, n, ~] = size(img);
    
    img_done = 0;
    
    key = [final_paths_crop{k, 2}, '_', final_paths_crop{k, 3}];
    if ~position_map.isKey(key)
        disp(sprintf('Key missmatch error %s', key)); 
        continue
    end
    
    pos = position_map(key);
    for i = 1:size(pos, 1)
       x = pos(i, 1);
       y = pos(i, 2);

       try
            img_crop = img((x-(output_size-1)):x, (y-(output_size-1)):y,:);
       catch
            fprintf('Error, our of bounds x:%d y:%d m:%d n:%d \n', ...
                [x, y, m, n]); 
            continue
       end

       img_done = img_done + 1;
       name = fullfile(save_path, [final_paths_crop{k, 2}, '_', ... 
            final_paths_crop{k, 3}, '_', num2str(img_done), '.tiff']);
       imwrite(img_crop, name);    
    end
    
    disp(sprintf('Cropped image: %d/%d \t crops: %d', ...
         [k, size(final_paths_crop, 1), img_done]));
end

% Local function for extracting the image paths
function final_paths = get_paths(data_path, sub_path)
    final_paths = {};

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
                if ~isempty(strfind(files(file).name, '.tiff'))
                   final_paths{size(final_paths, 1)+1, 1} = ...
                       fullfile(image_paths{image_nr}, files(file).name);
                   final_paths{size(final_paths, 1), 2} = sub_path{glas};
                   split = strsplit(image_paths{image_nr});
                   
                   id = split{end};
                   if strcmp(id, 'image')
                       id = split{end-3};
                   end
                   
                   final_paths{size(final_paths, 1), 3} = id;
                end
            end
        end
    end
end