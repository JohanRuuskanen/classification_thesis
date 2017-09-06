function ens_output = ...
    classifier_ensemble(db, idx, nu1_output, nu2_output, ...
    net_output, ens_output)
    % Function for fitting the final ensemble on the different classifiers.

    % Load the data sheet
    global data_sheet

    % Extract the different patients to sync the scores
    pat_nu1 = nu1_output(idx).pat_X;
    pat_nu1_v = nu1_output(idx).pat_X_v;
    
    pat_nu2 = nu2_output(idx).pat_X;
    pat_nu2_v = nu2_output(idx).pat_X_v;
    
    % Calculate the number of neural nets
    m = length(db(idx).img_sets);
    %m = 5;
    
    % Creating new score matrices
    scores = [0.5*ones(size(pat_nu1, 1), m)];
    scores_v = [0.5*ones(size(pat_nu1_v, 1), m)];
    
    %scores = [nu1_output(idx).scores, nu2_output(idx).scores ...
    %    0.5*ones(size(pat_nu1, 1), m)];
    %scores_v = [nu1_output(idx).scores_v, nu2_output(idx).scores_v ...
    %    0.5*ones(size(pat_nu1_v, 1), m)];  
    
    % Pair up the scores from the neural nets to the clinical data
    scores_net = cell2mat(net_output(idx).scores)';
    scores_net_v = cell2mat(net_output(idx).scores_v)';
    
    for j = 1:m
        pat_net = net_output(idx).pat{j}';
        pat_net_v = net_output(idx).pat_v{j}';

        % Run some tests to see that the different patients add up in the
        % different sets
        if ~isempty(setdiff(unique(pat_net), pat_nu1))
           disp('Warning! pat_net contains patients not included in pat_nu1'); 
        end   

        if ~isempty(setdiff(unique(pat_net_v), pat_nu1_v))
           disp('Warning! pat_net_v contains patients not included in pat_clin_valid'); 
        end

        for i = 1:length(pat_nu1)
            p = pat_nu1(i);

            s = mean(scores_net(pat_net == p, j));

            if ~isnan(s)
               scores(i, size(scores, 2) - m + j) = s; 
            end

        end

         for i = 1:length(pat_nu1_v)
            p = pat_nu1_v(i);

            s = mean(scores_net_v(pat_net_v == p, j));

            if ~isnan(s)
               scores_v(i, size(scores_v, 2) - m + j) = s; 
            end

         end
    end
    
    % Extract the features as the scores from the classfiers
    X = scores;
    Y = db(idx).Y;
    
    X_v = scores_v;
    Y_v = db(idx).Y_v;
    
    % Remove mean and std using the training set
    %mu = mean(X);
    %sig = std(X);
    
    %X = (X - mu) ./ sig;
    %X_v = (X_v - mu) ./ sig;
    
    % Fit logistic regression for ensemble
    
    %opt_options = struct;
    %opt_options.ShowPlots = 0;
    %opt_options.Verbose = 0;
    %opt_options.Repartition = 1;
    
    %{
    mdl = fitcsvm(X, Y, 'KernelFunction', 'linear', ...
        'Prior', 'uniform', ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', opt_options );
    
    %}
    %mdl = TreeBagger(500, X, Y, 'Prior', 'uniform');
    
    %mdl = fitcdiscr(X, Y, 'Prior', 'uniform', ...
    %    'OptimizeHyperparameters', 'auto',  ...
    %    'HyperparameterOptimizationOptions', opt_options);
    
    %[pred, final_score, ~] = predict(mdl, X);
    %[pred_v, final_score_v, ~] = predict(mdl, X_v);
    
    %pred = str2num(cell2mat(pred));
    %pred_v = str2num(cell2mat(pred_v));
    
    %final_score = final_score(:, 2);
    %final_score_v = final_score_v(:, 2);
    
    % Simple Voting
    final_score = mean(X, 2);
    final_score_v = mean(X_v, 2);
    
    pred = (final_score > 0.5) + 1;
    pred_v = (final_score_v > 0.5) + 1;
    
    [ROC_X, ROC_Y, ~, Auc_v] = perfcurve(Y_v, final_score_v, 2, ...
        'Prior', 'uniform');
    
    cm = confusionmat(Y, pred);
    cm = cm./repmat(sum(cm, 2), 1, size(cm, 2));
    
    cm_v = confusionmat(Y_v, pred_v);
    cm_v = cm_v./repmat(sum(cm_v, 2), 1, size(cm_v, 2));
    
    % Log results in struct
    ens_output(idx).Auc_v = Auc_v;
    ens_output(idx).CM = cm;
    ens_output(idx).CM_v = cm_v;
    
    ens_output(idx).X = X;
    ens_output(idx).Y = Y;
    
    ens_output(idx).ROC_X = ROC_X;
    ens_output(idx).ROC_Y = ROC_Y;
    
    ens_output(idx).final_score = final_score;
    ens_output(idx).final_score_v = final_score_v;
end

