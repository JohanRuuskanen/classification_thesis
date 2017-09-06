function [db] = extract_images(db, idx, data_paths)
    % Function for extracting the cropped images for each patient into
    % imagestore databases.

    global data_sheet
    pat = cell2mat(data_sheet(:, 4));
    
    % Extract the patients for label 0/1 and train/valid sets for partition
    % idx.
    train_0 = db(idx).train_0;
    valid_0 = db(idx).valid_0;
    
    train_1 = db(idx).train_1;
    valid_1 = db(idx).valid_1;

    % Save mismatches
    no_match = {};
    
    db(idx).img_sets = struct;
    % Find the corresponding images
    for j = 1:length(data_paths)
        train_0_files = find_images(train_0, pat, data_paths{j});
        train_0_labels = zeros(size(train_0_files)) + 1;
        valid_0_files = find_images(valid_0, pat, data_paths{j});
        valid_0_labels = zeros(size(valid_0_files)) + 1;

        train_1_files = find_images(train_1, pat, data_paths{j});
        train_1_labels = ones(size(train_1_files)) + 1;  
        valid_1_files = find_images(valid_1, pat, data_paths{j});
        valid_1_labels = ones(size(valid_1_files)) + 1;

        % check if files exists, if not remove. We still keep the 
        % numerical data for that patient if the file does not exist.
        [train_0_files, train_0_labels, no_match] = ...
            file_exists(train_0_files, train_0_labels, no_match);
        [valid_0_files, valid_0_labels, no_match] = ...
            file_exists(valid_0_files, valid_0_labels, no_match);

        [train_1_files, train_1_labels, no_match] = ...
            file_exists(train_1_files, train_1_labels, no_match);
        [valid_1_files, valid_1_labels, no_match] = ...
            file_exists(valid_1_files, valid_1_labels, no_match);

        % Create ImageDatastore objects to save 
        train_set = imageDatastore(data_paths{j}); 
        valid_set = imageDatastore(data_paths{j});

        train_set.Files = {train_0_files{:}, train_1_files{:}};
        train_set.Labels = categorical([train_0_labels, train_1_labels]);

        valid_set.Files = {valid_0_files{:}, valid_1_files{:}};
        valid_set.Labels = categorical([valid_0_labels, valid_1_labels]);

        db(idx).img_sets(j).train = shuffle(train_set); 
        db(idx).img_sets(j).valid = shuffle(valid_set); 
    end
     
     % Test databases
    disp('Starting image tests')
    disp(sprintf('Number of mismatches: %d', [length(no_match)]));
    no_match
    for j = 1:length(data_paths)
        train_set = db(idx).img_sets(j).train;
        valid_set = db(idx).img_sets(j).valid;

        tbl_train = train_set.countEachLabel;
        tbl_valid = valid_set.countEachLabel;

        tbl_train = table2array(tbl_train(:, 2));
        tbl_valid = table2array(tbl_valid(:, 2));


        fprintf('Type \t\t class 0 \t class 1\n')
        fprintf('train img \t %d \t\t %d\n', ...
            [tbl_train(1), tbl_train(2)])
        fprintf('valid img \t %d \t\t %d\n', ...
            [tbl_valid(1), tbl_valid(2)]);
        fprintf('total img: %d \t train percentage: %.2f\n', ...
            [tbl_train(1) + tbl_train(2) ...
            + tbl_valid(1) + tbl_valid(2), ...
            length(train_set.Labels) / (length(train_set.Labels) + ...
            length(valid_set.Labels))])

        train_files = train_set.Files;
        valid_files = valid_set.Files;

        if length(train_files) ~= length(unique(train_files))
           disp('Error: Train set elements are not unique') 
        end
        if length(valid_files) ~= length(unique(valid_files))
           disp('Error: valid set elements are not unique') 
        end

        if length(intersect(train_files, valid_files)) > 0
            disp('Error: train and valid sets are not disjunct');
        end
    end
    disp('Tests complete')
    
end

function files = find_images(patient_set, pat, data_path)
    % Subfunction for finding the correct images

    global data_sheet
    files = {};
    for pat_nr = 1:length(patient_set)
        patient = patient_set(pat_nr);

        % Read Glas, IDs and labels for .mat file
        patient_info = data_sheet(pat == patient, :);
        
        glas = patient_info(:, 2);
        glas = char(glas{1});
        ids = patient_info(:, 3);
        
        % Include marker
        include = cell2mat(patient_info(:, 6)) == 0;

        for id_nr = 1:length(ids)
            if include(id_nr) == 0
                continue; 
            end
            
            id = ids{id_nr};
            
            prefix = fullfile(data_path, [glas, '_', num2str(id)]);
            img_files = dir([prefix, '_*']);
            
            if length(img_files) == 0
               img_files = dir([prefix, '.tiff']);
            end

            for img_nr = 1:length(img_files)
                img_name = img_files(img_nr).name;
                files{length(files)+1} = ...
                    fullfile(data_path, img_name); 
            end
        end
        
    end
end

function [files, labels, no_match] =  file_exists(files, labels, no_match)
    % Function for checking so that the image files actually exists
    
    file_nr = 1;
    while file_nr <= length(files)
        file = files{file_nr};
        if exist(file, 'file') ~= 2
            files(file_nr) = [];
            labels(file_nr) = [];

            if length(strmatch(file, no_match)) == 0
               no_match{length(no_match) + 1} = file; 
            end
        else
           file_nr = file_nr + 1; 
        end
    end
end

