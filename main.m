% Main script for controlling the entire program.
%% Set parameters

clear all
close all

% Load the data sheet and make it global
global data_sheet
load data3.mat

% Add path to dimensional reduction toolbox, visit
% https://lvdmaaten.github.io/drtoolbox/ for more information
addpath(genpath('../drtoolbox'));

% The number of folds
k = 5;

% Options for the neural nets
net_options = struct; 
net_options.MaxEpochs = 75;
net_options.LRdrop = 0.9;
net_options.LRperiod = 2;
net_options.LRinit = 0.001;

% Define data paths for the crops
data_paths = {'../data', '../data_1'};

%% Partition data and create output structs

% Partition data with k-fold
db = partition_data(k, 3, 0);

% Output structs
nu1_output = struct;
nu2_output = struct;
net_output = struct;
ens_output = struct;

% Save the partition. Dont uncomment this! We want the same partition
%save('partition.mat', 'db');

%% Run the machinery and pray it doesn't crash after a day

% Load the state to keep the same partitions and old computations. Comment
% this out if this is your first run.
%load state.mat

% If you want to restart with a saved partition, decomment the following.
load partition.mat

% First, define which fold you want to train on. 1 <= idx <= k
for idx = 1:5

    % Extract the numerical data
    db = extract_numerical(db, idx);

    % Extract the images from the data paths and patition into the database
    db = extract_images(db, idx, data_paths);

    fprintf('\nStarting ensemble training\n')
    tic

    fprintf('\nRunning Numerical Classifier on Prognosis Data\n')
    nu1_output = classifier_numerical_1(db, idx, nu1_output);

    fprintf('\nRunning Numerical Classifier on Extracted Image Data\n')
    nu2_output = classifier_numerical_2(db, idx, nu2_output);

    fprintf('\nRunning CNN Classifiers\n')

    m = length(data_paths);

    % Initiate neural net variabels
    net_output(idx).trainAcc = cell(m, 1);
    net_output(idx).validAcc = cell(m, 1);
    net_output(idx).voteAcc = cell(m, 1);
    net_output(idx).CM = cell(m, 1);
    net_output(idx).CM_v = cell(m, 1);
    net_output(idx).CM_vote = cell(m, 1);
    net_output(idx).Auc_v = cell(m, 1);
    net_output(idx).Auc_vote = cell(m, 1);

    net_output(idx).pat = cell(m, 1);
    net_output(idx).pat_v = cell(m, 1);   

    net_output(idx).scores = cell(m, 1);
    net_output(idx).scores_v = cell(m, 1);

    % Run the network seperately 5 times to save each network should it crash
    for j = 1:m

        fprintf('Training network %d/%d\n', j, m)

        % Clear the graphic card memory
        gpuDevice(1);

        % Run the Network
        net_output = classifier_neuralnet(db, idx, j, net_output, net_options);

        save('state_tmp.mat')
    end

    fprintf('training completed. Running time: %d m %d s\n', ...
        [floor(toc/60), round(mod(toc, 60))]);
  
end

save('state.mat')

%% Run the ensemble and get a single classification from the models

idx = 1;
ens_output = classifier_ensemble(db, idx, nu1_output, nu2_output, ...
    net_output, ens_output);

evaluate_model(db, idx, nu1_output, nu2_output, net_output, ens_output);