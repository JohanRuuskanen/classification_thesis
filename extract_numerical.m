function [db] = extract_numerical(db, idx)
    % Function for extracting the numerical data for each patient
    
    global data_sheet
    
    pat = cell2mat(data_sheet(:, 4));
    
    % Extraxt unique patients
    [pat_unique, location] = unique(pat); 
    data_sheet_unique = data_sheet(location, :);
    
    % Extract the patients for label 0/1 and train/valid sets for partition
    % idx.
    train_0 = db(idx).train_0;
    valid_0 = db(idx).valid_0;
    
    train_1 = db(idx).train_1;
    valid_1 = db(idx).valid_1;

    % Find the corresponding numerical data
    [X, Y, pat_X] =  ...
        find_num_data(data_sheet_unique, train_0, train_1);
    [X_v, Y_v, pat_X_v] = ...
        find_num_data(data_sheet_unique, valid_0, valid_1);

    % Shuffle the numerical data
    ind = randsample(1:length(Y), length(Y));
    ind_v = randsample(1:length(Y_v), length(Y_v));

    % Save the data to the database
    db(idx).pat_X = pat_X(ind);
    db(idx).pat_X_v = pat_X_v(ind_v);

    db(idx).X = X(ind, :);
    db(idx).Y = Y(ind);
    db(idx).X_v = X_v(ind_v, :);
    db(idx).Y_v = Y_v(ind_v);
    
    % Start numerical tests
    disp('Starting tests')
   
    X = db(idx).X;
    X_v = db(idx).X_v;

    Y = db(idx).Y;
    Y_v = db(idx).Y_v;

    fprintf('\nFold %d\n', [idx]);
    fprintf('Type \t\t class 0 \t class 1\n')
    fprintf('train num \t %d \t\t %d\n', ...
        [sum(Y == 1), sum(Y == 2)]);
    fprintf('valid num \t %d \t\t %d\n', ...
        [sum(Y_v == 1), sum(Y_v == 2)]);
    fprintf('total num: %d \t\t train percentage: %.2f\n', ...
        [length(Y) + length(Y_v), length(Y) / (length(Y) + length(Y_v))]);

    if sum(sum(isnan([X; X_v]))) > 0 || sum(isnan([Y; Y_v])) > 0 
        disp('Error: The numerical data contain NaN values')
    end

    disp('Tests complete')
    
end

function [X, Y, pat] = find_num_data(ds, idx_0, idx_1)
    % Subfunction for actually reading the data file
    
    global data_sheet

    idx = [idx_0; idx_1];
    
    % Extract the clinical data
    % Need patient number for ensemble connection
    pat = cell2mat(ds(idx, 4));

    % Primary variables
    age = cell2mat(ds(idx, 11));
    tum_size = cell2mat(ds(idx,13));
    ds(278:end,15) = {0};
    lymf_met = cell2mat(ds(idx,15));

    % Add other variables
    % er_ihc
    er_ihc = cellstr(ds(idx,23));
    er_ihc = replace(er_ihc,'Ja','1');
    er_ihc = str2double(replace(er_ihc,'Nej','0'));
    er_ihc((er_ihc == 0) + (er_ihc == 1) == 0) = 0.5;

    % pgr_ihc
    pgr_ihc = cellstr(ds(idx,25));
    pgr_ihc = replace(pgr_ihc,'Ja','1');
    pgr_ihc = str2double(replace(pgr_ihc,'Nej','0'));
    pgr_ihc((pgr_ihc == 0) + (pgr_ihc == 1) == 0) = 0.5;

    % her2
    her2 = cellstr(ds(idx,26));
    her2 = replace(her2,'Ja','1');
    her2 = str2double(replace(her2,'Nej','0'));
    her2((her2 == 0) + (her2 == 1) == 0) = 0.5;

    % ki67
    ki67 = cellstr(ds(idx,27));
    ki67 = replace(ki67,'Ja','1');
    ki67 = str2double(replace(ki67,'Nej','0'));
    ki67((ki67 == 0) + (ki67 == 1) == 0) = 0.5;

    % hist grad
    hist_grad = ds(idx, 28);
    for i = 1:length(hist_grad)
        if isempty(hist_grad{i})
            hist_grad{i} = 0;
        end
    end
    hist_grad = cell2mat(hist_grad);

    X_1 = [age, tum_size, lymf_met, er_ihc, pgr_ihc, her2, ki67, hist_grad];

    % Extract numerical data
    X_2 = ds(idx, 36:end);
    
    % Recheck the data_sheet to include numerical data removed by unique by
    % mistake
    pat_orig = cell2mat(data_sheet(:, 4));
    for i = 1:size(X_2, 1)
        if isempty(X_2{i, 1})
            p = idx(i);

            num_data = data_sheet(pat_orig == p, 36:end);
    
            for j = 1:size(num_data, 1)
               if ~isempty(num_data{j, 1})
                   X_2(i, :) = num_data(j, :);
                   break;
               end
           end
        end
    end
    
    % See which patients that has no entries
    include = cellfun('isempty', X_2(:, 1)) == 0;
    
    inc_0 = include(ismember(idx, idx_0));
    inc_1 = include(ismember(idx, idx_1));    

    X = [X_1(include, :), cell2mat(X_2(include, :))];
    
    Y = [zeros(sum(inc_0), 1) + 1; ones(sum(inc_1), 1) + 1];
    
    pat = pat(include);
    
end
