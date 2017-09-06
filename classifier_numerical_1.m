function output = classifier_numerical_1(db, idx, output)
    % Function for fitting the numerical classifiers on the clinical data.

    % Number of classifiers
    m = 4;
    
    % Extract the partition and the clinical data
    X = db(idx).X(:, 1:8);
    Y = db(idx).Y;
    
    X_v = db(idx).X_v(:, 1:8);
    Y_v = db(idx).Y_v;
    
    % Remove mean and std using the training set
    mu = mean(X);
    sig = std(X);
    
    X = (X - mu) ./ sig;
    X_v = (X_v - mu) ./ sig;
    
    % options for hyperparameter optimization, needed for SVM and Fischers
    % Linear Discriminant. 
    opt_options = struct;
    opt_options.ShowPlots = 0;
    opt_options.Verbose = 0;
    opt_options.Repartition = 1;
    
    % Fit models
    fprintf('Fitting linear SVM\n');
    svm_model_lin = fitcsvm(X, Y, 'KernelFunction', 'linear', ...
        'Prior', 'uniform', ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', opt_options );
    
    fprintf('Fitting RBF SVM\n');
    svm_model_rbf = fitcsvm(X, Y, 'KernelFunction', 'rbf', ...
        'Prior', 'uniform',...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', opt_options);
    
    fprintf('Fitting Random Forest\n');
    rf_model = TreeBagger(500, X, Y, 'Prior', 'uniform');
    
    fprintf('Fitting FLD\n');
    fld_model = fitcdiscr(X, Y, 'Prior', 'uniform', ...
        'OptimizeHyperparameters', 'auto',  ...
        'HyperparameterOptimizationOptions', opt_options);
    
    % Initiate output variables
    Acc = zeros(m, 1);
    Acc_v = zeros(m, 1);
    Auc_v = zeros(m, 1);
    ROC = cell(m, 2);
    CM_v = zeros(2, 2, m);
    scores = zeros(size(X, 1), m);
    scores_v = zeros(size(X_v, 1), m); 
    
    fprintf('Calculating scores\n');
    % Test models, compare accuracy to catch overfitting
    [scores(:, 1), scores_v(:, 1), Acc(1), ...
        Acc_v(1), Auc_v(1), ROC(1, :), CM_v(:, :, 1)] = ...
        evaluate_num_model(svm_model_lin, X, Y, X_v, Y_v);
    [scores(:, 2), scores_v(:, 2), Acc(2),  ...
        Acc_v(2), Auc_v(2), ROC(2, :), CM_v(:, :, 2)] = ...
        evaluate_num_model(svm_model_rbf, X, Y, X_v, Y_v);
    [scores(:, 3), scores_v(:, 3), Acc(3), ...
        Acc_v(3), Auc_v(3), ROC(3, :), CM_v(:, :, 3)] = ...
        evaluate_num_model(rf_model, X, Y, X_v, Y_v);
    [scores(:, 4), scores_v(:, 4), Acc(4), ...
        Acc_v(4), Auc_v(4), ROC(4, :), CM_v(:, :, 4)] = ...
        evaluate_num_model(fld_model, X, Y, X_v, Y_v);
    
    % Save the output variables to the output struct
    output(idx).Acc = Acc;
    output(idx).Acc_v = Acc_v;
    output(idx).Auc_v = Auc_v;
    output(idx).CM_v = CM_v;
    output(idx).ROC = ROC;
    
    output(idx).scores = scores;
    output(idx).scores_v = scores_v;
    
    output(idx).pat_X = db(idx).pat_X;
    output(idx).pat_X_v = db(idx).pat_X_v;
end

function [score, score_v, Acc, Acc_v, Auc_v, ROC, CM_v] =  ...
    evaluate_num_model(mdl, X, Y, X_v, Y_v)
    % Calculates confusion matrix for train and valid set and AUC score for 
    % the validation set of the input model mdl.

    % Predict the training and validation data
    [pred, score, ~] = predict(mdl, X);
    [pred_v, score_v, ~] = predict(mdl, X_v);
    
    score = score(:, 2);
    score_v = score_v(:, 2);
    
    % Some models writes outputs as character arrays for some odd reason
    try
        pred = str2num(cell2mat(pred));
        pred_v = str2num(cell2mat(pred_v));
    catch
    end
    
    ROC = cell(1, 2);
    % Calculate the AUC score of the validation prediciton
    [ROC{1, 1}, ROC{1, 2}, ~, Auc_v] = perfcurve(Y_v, score_v, 2, ...
        'Prior', 'uniform');

    % Calculate the confusion matrix and total uniform weighted accuracy 
    % for the training and validation data 
    confMat = confusionmat(Y, pred);
    confMat = confMat./repmat(sum(confMat, 2), 1, 2);
    Acc = (confMat(1)+confMat(4))/2;
    
    confMat_v = confusionmat(Y_v, pred_v);
    confMat_v = confMat_v./repmat(sum(confMat_v, 2), 1, 2);
    Acc_v = (confMat_v(1)+confMat_v(4))/2;
    
    CM_v = confMat_v;
    
end

