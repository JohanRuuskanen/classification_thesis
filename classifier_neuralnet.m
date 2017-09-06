function net_output = classifier_neuralnet(db, idx, j, net_output, net_options)
    % Function for fitting the nerual net to the image data

    global data_sheet
    
    % Extract a single instance from the k-fold
    train_set = db(idx).img_sets(j).train;
    valid_set = db(idx).img_sets(j).valid;

    % Calculate channel normalization from the training set
    fprintf('Calculating normalization\n');
    channel_mean = normalization(train_set);

    % Change the read-in function
    train_set.ReadFcn = @(filename) read_and_preprocess_image(filename, ...
        channel_mean);
    valid_set.ReadFcn = @(filename) read_and_preprocess_image(filename, ...
        channel_mean);

    % Load the network of create your own!
   
    % Alexnet
    %{
    convnet = alexnet();
    layers = convnet.Layers;

    layers(1) = imageInputLayer([227 227 3], 'DataAugmentation', 'None', ...
    layers(23) = fullyConnectedLayer(2, 'Name', 'fc8');
    layers(25) = classificationLayer('Name', 'ClassificationLayer');
    %}
    
    % AlexNet smaller FC layers
    %{
    convnet = alexnet();
    layers = convnet.Layers;

    layers(1) = imageInputLayer([227 227 3], 'Name', 'input', ...
        'DataAugmentation', 'None');
    layers(17) = fullyConnectedLayer(4096, 'Name', 'fc6');
    layers(20) = fullyConnectedLayer(1024, 'Name', 'fc7');
    layers(23) = fullyConnectedLayer(2, 'Name', 'fc8');
    layers(25) = classificationLayer('Name', 'ClassificationLayer');
    %}
    
    % AlexNet randomize all layers
    %{
    convnet = alexnet();
    layers = convnet.Layers;

    layers(1) = imageInputLayer([227 227 3], 'Name', 'input', ...
        'DataAugmentation', 'None');
    layers(2) = convolution2dLayer(11, 96, 'stride', 4, 'padding', 0, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv1');
    layers(6) = convolution2dLayer(5, 256, 'stride', 1, 'padding', 2, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv2');
    layers(10) = convolution2dLayer(3, 384, 'stride', 1, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv3');
    layers(12) = convolution2dLayer(3, 384, 'stride', 1, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv4');
    layers(14) = convolution2dLayer(3, 256, 'stride', 1, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv5');   
    layers(17) = fullyConnectedLayer(4096, 'Name', 'fc6');
    layers(20) = fullyConnectedLayer(4096, 'Name', 'fc7');
    layers(23) = fullyConnectedLayer(2, 'Name', 'fc8');
    layers(25) = classificationLayer('Name', 'ClassificationLayer');
    %}
    
    %{
    %Smallnet-4
    layers = [imageInputLayer([227 227 3], 'Name', 'input', ...
        'Normalization', 'none'); 
        convolution2dLayer(5, 32, 'stride', 3, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv1');     
        reluLayer('Name','relu1');
        convolution2dLayer(3, 64, 'stride', 2, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv2');
        reluLayer('Name','relu2');
        fullyConnectedLayer(256, 'BiasLearnRateFactor', 2, ...
            'WeightL2Factor', 1, 'Name','fc3');
        dropoutLayer(0.5);
        reluLayer('Name','relu3');
        fullyConnectedLayer(length(unique(train_set.Labels)), ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1,  'Name','fc4');
        softmaxLayer('Name','prob');
        classificationLayer('Name','output')];
    %}
    
    %Smallnet-5
    layers = [imageInputLayer([227 227 3], 'Name', 'input', ...
        'Normalization', 'none'); 
        convolution2dLayer(5, 32, 'stride', 3, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv1');     
        reluLayer('Name','relu1');
        convolution2dLayer(3, 64, 'stride', 1, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv2');
        reluLayer('Name','relu2');
         convolution2dLayer(3, 64, 'stride', 2, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv3');
        reluLayer('Name','relu3');
        fullyConnectedLayer(256, 'BiasLearnRateFactor', 2, ...
            'WeightL2Factor', 1, 'Name','fc4');
        dropoutLayer(0.5);
        reluLayer('Name','relu4');
        fullyConnectedLayer(length(unique(train_set.Labels)), ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1,  'Name','fc5');
        softmaxLayer('Name','prob');
        classificationLayer('Name','output')];

    %{
    %SmallNet-6
    layers = [imageInputLayer([227 227 3], 'Name', 'input', ...
        'Normalization', 'none'); 
        convolution2dLayer(5, 64, 'stride', 3, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv1');     
        reluLayer('Name','relu1');
        convolution2dLayer(3, 128, 'stride', 2, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv2');
        reluLayer('Name','relu2');
        maxPooling2dLayer(2, 'stride', 2, 'padding', 0, 'name', 'pool2');
        convolution2dLayer(3, 128, 'stride', 1, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv3');
        reluLayer('Name','relu3');
        convolution2dLayer(3, 128, 'stride', 1, 'padding', 1, ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1, 'Name','conv4');
        reluLayer('Name','relu4');
        maxPooling2dLayer(2, 'stride', 2, 'padding', 0, 'name', 'pool4');
        fullyConnectedLayer(256, 'BiasLearnRateFactor', 2, ...
            'WeightL2Factor', 1, 'Name','fc5');
        dropoutLayer(0.5);
        reluLayer('Name','relu5');
        fullyConnectedLayer(length(unique(train_set.Labels)), ...
            'BiasLearnRateFactor', 2, 'WeightL2Factor', 1,  'Name','fc6');
        softmaxLayer('Name','prob');
        classificationLayer('Name','output')];
    %}
    
    % Extract the parameters
    MaxEpochs = net_options.MaxEpochs;
    LRdrop = net_options.LRdrop;
    LRperiod = net_options.LRperiod;
    LRinit = net_options.LRinit;

    trainAcc = []; 
    validAcc = []; 
    voteAcc = [];
    
    % Some uglyness since MATLAB 2017a implementation of CNN can't 
    % calculate validation accuracy on-line, also because we need to
    % repartition the training set to get balanced classes.
    fprintf('Training Network, this part may take several hours\n');
    for k = 1:MaxEpochs

        % Redefine the training options to change the learning rate LR
        LR = LRinit*LRdrop^(floor(k/LRperiod));
        options = trainingOptions('sgdm', ...
            'MaxEpochs', 1, ...
            'InitialLearnRate', LR, ...
            'MiniBatchSize', 256, ...
            'Verbose', 0, ...
            'VerboseFrequency', 1, ...
            'ExecutionEnvironment', 'gpu');

        % Repartioning the training set to include the same number of classes
        tbl = countEachLabel(train_set);
        minSetCount = min(tbl{:, 2});
        train_set_part = splitEachLabel(train_set, minSetCount, 'randomize'); 

        % Train the network 1 epoch
        [trainedNet, trainInfo] = trainNetwork(train_set_part, ...
            layers, options);

        % Accuracy and confMat for train
        [predictedLabels_train, ~] = ...
            classify(trainedNet, train_set_part);

        confMat_train = confusionmat(train_set_part.Labels, ...
            predictedLabels_train);
        confMat_train = confMat_train./repmat(sum(confMat_train, 2), 1, ...
            size(confMat_train, 2));

        % Accuracy and confMat for valid
        [predictedLabels_valid, score_valid] = ...
            classify(trainedNet, valid_set);

        confMat_valid = confusionmat(valid_set.Labels, predictedLabels_valid);
        confMat_valid = confMat_valid./repmat(sum(confMat_valid, 2), 1, ...
            size(confMat_valid, 2));

        % Accuracy for Voting system for valid set
        confMat_vote = image_vote(valid_set, score_valid(:, 2));

        % Save accuracies
        trainAcc = [trainAcc, sum(diag(confMat_train))/2];
        validAcc = [validAcc, sum(diag(confMat_valid))/2];
        voteAcc = [voteAcc, sum(diag(confMat_vote))/2];

        layers = trainedNet.Layers;
        reset(train_set); reset(valid_set);
    end
    
    fprintf('Final classification and vote score calculation\n');
    % Rerun the score classification on the entire training set
    [pl_train, score] =  classify(trainedNet, train_set);
    
    confMat_train = confusionmat(train_set.Labels, pl_train);
    confMat_train = confMat_train./repmat(sum(confMat_train, 2), 1, ...
        size(confMat_train, 2));

    % Calculate the results on the validation set
    [pl_valid, score_valid] =  classify(trainedNet, valid_set);
    
    confMat_valid = confusionmat(valid_set.Labels, pl_valid);
    confMat_valid = confMat_valid./repmat(sum(confMat_valid, 2), 1, ...
        size(confMat_valid, 2));
    
    % Voting to get a single result per patient
    [~, score_vote, ~,  pat] = image_vote(train_set, score(:, 2));
    [confMat_vote, score_vote_valid, vote_correct, pat_v] = ...
        image_vote(valid_set, score_valid(:, 2));
    
    % Calculate ROC and AUC
    [ROC_X, ROC_Y, ~, Auc_v] = perfcurve(db(idx).img_sets(j).valid.Labels, ...
        score_valid(:, 2), 2, 'Prior', 'uniform');
    [ROC_X_vote, ROC_Y_vote, ~, Auc_vote] = perfcurve(vote_correct+1, ...
        score_vote_valid, 2, 'Prior', 'uniform');
    
    % Log results in output struct
    net_output(idx).trainAcc{j} = trainAcc;
    net_output(idx).validAcc{j} = validAcc;
    net_output(idx).voteAcc{j} = voteAcc;
    net_output(idx).CM{j} = confMat_train;
    net_output(idx).CM_v{j} = confMat_valid;
    net_output(idx).CM_vote{j} = confMat_vote;
    net_output(idx).Auc_v{j} = Auc_v;
    net_output(idx).Auc_vote{j} = Auc_vote;
    
    net_output(idx).pat{j} = pat;
    net_output(idx).pat_v{j} = pat_v;    
    
    net_output(idx).ROC_X{j} = ROC_X;
    net_output(idx).ROC_Y{j} = ROC_Y; 
    
    net_output(idx).ROC_X_vote{j} = ROC_X_vote;
    net_output(idx).ROC_Y_vote{j} = ROC_Y_vote; 
    
    net_output(idx).scores{j} = score_vote;
    net_output(idx).scores_v{j} = score_vote_valid;
    
end

function [channel_mean] = normalization(data_set)
    
    % Create empty variable
    channel_mean = reshape([0, 0, 0], 1, 1, 3);
    
    % Loop through each image in the data set
    for i = 1:length(data_set.Files)
        
        % Read the image
        img = double(imread(data_set.Files{i}));
        
        % Extract the sizes
        [m, n, ~] = size(img);
        
        % Calculate the mean for that image in each channel
        img_mean = sum(sum(img)) / (m*n);
        
        % Add the image channel mean to the total channel mean
        channel_mean = channel_mean + img_mean;
    end
    
    % Finalize the mean by dividing by the number of images used
    channel_mean = channel_mean / length(data_set.Files);
    
end
