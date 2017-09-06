function [db] = partition_data(k, to_use, norm_class)
    % Function for partitioning the patients into a k-fold cross-validation

    % Load patients and labels from .mat file
    global data_sheet
    pat = cell2mat(data_sheet(:, 4));

    % Which data sets to use
    if to_use == 1
        pat = pat(pat <= 227);
    elseif to_use == 2
        pat = pat(pat > 227);
    end
    
    [pat_unique, location] = unique(pat); 
    data_sheet_unique = data_sheet(location, :);
    labels = cell2mat(data_sheet_unique(:, 30));
    
    % Split patients into the labeled groups
    pat_0 = pat_unique(labels == 0);
    pat_1 = pat_unique(labels == 1);
    
    % Partition into K splits
    c_0 = cvpartition(pat_0, 'kfold', k);
    c_1 = cvpartition(pat_1, 'kfold', k);
    db = struct;
    
    fprintf('\n');

    for i = 1:k
        disp(sprintf('Creating partition %d/%d', [i, k]));
        
        train_0 = pat_0(c_0.training(i));
        valid_0 = pat_0(c_0.test(i));

        train_1 = pat_1(c_1.training(i));
        valid_1 = pat_1(c_1.test(i));
        
        % Resample so that train/valid sets have the equal amount of
        % patients. Only if norm_class is set to True.
        if norm_class
            ml = min(length(train_0), length(train_1));
            train_0 = randsample(train_0, ml);
            train_1 = randsample(train_1, ml);
            
            ml = min(length(valid_0), length(valid_1));
            valid_0 = randsample(valid_0, ml);
            valid_1 = randsample(valid_1, ml);        
        end
    end

    db(i).train_0 = train_0;
    db(i).valid_0 = valid_0;

    db(i).train_1 = train_1;
    db(i).valid_1 = valid_1;
    
end
