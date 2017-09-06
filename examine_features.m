function [X_pca, X_tsne, idx] = examine_features(X, Y)
    % Computes the PCA and t-SNE reductions of the data and then plots the
    % feature points in two subplots. Also plots the singular values from
    % the PCA
    % Returns the reductions
    
    % Calculate reductions
    [X_pca, model_pca] = pca(X, size(X, 2));
    X_tsne = tsne(X);

    % Plot handling
    classes = unique(Y);
    c = cell(length(classes), 1);
    for k = 1:length(classes)
        c{k} = Y == classes(k); 
    end

    figure(1)
    subplot(121)
    hold on;
    for k = 1:length(c)
        scatter(X_pca(c{k}, 1), X_pca(c{k}, 2), 'o')
    end
    title('PCA')
    subplot(122)
    hold on;
    for k = 1:length(c)
        scatter(X_tsne(c{k}, 1), X_tsne(c{k}, 2), 'o')
    end
    title('t-SNE')
    
    figure(2)
    plot(model_pca.lambda, 'o')
    xlabel('Principal component')
    ylabel('Strength')
    
    
    % Calculate the variability of the different classes/features    
    X_0 = X(Y == 1, :);
    X_1 = X(Y == 2, :);
    
    n_0 = size(X_0, 1);
    n_1 = size(X_1, 1);
    
    mu_0 = mean(X_0);
    mu_1 = mean(X_1);
    
    sig_0 = std(X_0);
    sig_1= std(X_1);
    
    figure(3)
    hold on;
    plot(mu_0, 'b*')
    plot(mu_1, 'r*')
    legend('Class 1', 'Class 2')
    title('Mean values')
    
    upper_0 = mean(X_0) + 1.96*sig_0/sqrt(n_0);
    lower_0 = mean(X_0) - 1.96*sig_0/sqrt(n_0);
    
    upper_1 = mean(X_1) + 1.96*sig_1/sqrt(n_1);
    lower_1 = mean(X_1) - 1.96*sig_1/sqrt(n_1);
    
    overlap = zeros(size(X_0, 2), 1);
    for i = 1:size(X_0, 2);
        u0 = upper_0(i);
        l0 = lower_0(i);
        
        u1 = upper_1(i);
        l1 = lower_1(i);
        
        if u0 > u1 && u1 > l0
            overlap(i) = 1;
        elseif u0 > l1 && l1 > l0
            overlap(i) = 1;
        end
    end
    
    idx = overlap == 0;
    
    figure(4)
    plot(abs(mu_0 - mu_1), '*')
    title('Mean Difference')
    
end

