function [] = evaluate_model(db, idx, nu1_output, nu2_output, net_output, ens_output)
    % Function for handling all the result presentations.

    close all;
    
    
    fprintf('\n-----BEGINNING NEW EVALUATION-----\n')
    fprintf('NUMERICAL 1 EVAULATION\n')
    fprintf('1. SVM - Lin \t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu1_output(idx).Acc(1), nu1_output(idx).Acc_v(1), nu1_output(idx).Auc_v(1)])
     fprintf('2. SVM - RBF \t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu1_output(idx).Acc(2), nu1_output(idx).Acc_v(2), nu1_output(idx).Auc_v(2)])
     fprintf('3. RF \t\t\t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu1_output(idx).Acc(3), nu1_output(idx).Acc_v(3), nu1_output(idx).Auc_v(3)])
     fprintf('4. FLD \t\t\t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu1_output(idx).Acc(4), nu1_output(idx).Acc_v(4), nu1_output(idx).Auc_v(4)])
    
    fprintf('Confusion Matrix valid\n') 
    nu1_output(idx).CM_v
    
    fprintf('NUMERICAL 2 EVAULATION\n')
    fprintf('1. SVM - Lin \t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu2_output(idx).Acc(1), nu2_output(idx).Acc_v(1), nu2_output(idx).Auc_v(1)])
     fprintf('2. SVM - RBF \t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu2_output(idx).Acc(2), nu2_output(idx).Acc_v(2), nu2_output(idx).Auc_v(2)])
     fprintf('3. RF \t\t\t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu2_output(idx).Acc(3), nu2_output(idx).Acc_v(3), nu2_output(idx).Auc_v(3)])
     fprintf('4. FLD \t\t\t Train Acc: %0.2f | valid acc: %0.2f | AUC: %0.2f \n', ...
        [nu2_output(idx).Acc(4), nu2_output(idx).Acc_v(4), nu2_output(idx).Auc_v(4)])
    
    fprintf('Confusion Matrix valid\n') 
    nu2_output(idx).CM_v
    
    fprintf('NETWORK EVALUATION\n')
    for j = 1:length(net_output(idx).Auc_v)
        fprintf('%d. AUC valid: %0.2f | AUC vote: %0.2f\n', ...
            [j, net_output(idx).Auc_v{j}, net_output(idx).Auc_vote{j}])
    end
    
    for j = 1:length(net_output(idx).trainAcc)
        fprintf('Confusion matrices for network %d\n', [j])
        
        fprintf('Train\n') 
        net_output(idx).CM{j}

        fprintf('Valid\n') 
        net_output(idx).CM_v{j}
        
        fprintf('Vote\n') 
        net_output(idx).CM_vote{j}
    end
    
    fprintf('ENSEMBLE EVALUATION\n')
    fprintf('AUC score valid: %0.2f\n', [ens_output(idx).Auc_v])
    
    fprintf('Confusion Matrix train\n') 
    ens_output(idx).CM
    
    fprintf('Confusion Matrix valid\n') 
    ens_output(idx).CM_v
    
    
    figure()
    title('ROC numerical 1')
    hold on;
    plot(nu1_output(idx).ROC{1, 1}, nu1_output(idx).ROC{1, 2});
    plot(nu1_output(idx).ROC{2, 1}, nu1_output(idx).ROC{2, 2});
    plot(nu1_output(idx).ROC{3, 1}, nu1_output(idx).ROC{3, 2});
    plot(nu1_output(idx).ROC{4, 1}, nu1_output(idx).ROC{4, 2});
    
    plot([0, 1], [0, 1], 'k--');
    legend('SVM Lin', 'SVM RBF', 'RF', 'FLD', 'Location', 'southeast')
    
    
    figure()
    title('ROC numerical 2')
    hold on;
    plot(nu2_output(idx).ROC{1, 1}, nu2_output(idx).ROC{1, 2});
    plot(nu2_output(idx).ROC{2, 1}, nu2_output(idx).ROC{2, 2});
    plot(nu2_output(idx).ROC{3, 1}, nu2_output(idx).ROC{3, 2});
    plot(nu2_output(idx).ROC{4, 1}, nu2_output(idx).ROC{4, 2});
    
    plot([0, 1], [0, 1], 'k--');
    legend('SVM Lin', 'SVM RBF', 'RF', 'FLD', 'Location', 'southeast')
    
    figure()
    title('ROC net valid')
    hold on;
    for j = 1:length(net_output(idx).ROC_X)
        plot(net_output(idx).ROC_X{j}, net_output(idx).ROC_Y{j});
    end
    plot([0, 1], [0, 1], 'k--');
    legend('All Colors', 'Data 1', 'Data 2', 'Data 3', 'Data 4', 'Data 5', ...
        'Location', 'southeast')
    xlabel('False positive rate')
    ylabel('True positive rate');
    
    figure()
    title('ROC net vote')
    hold on;
    for j = 1:length(net_output(idx).ROC_X_vote)
        plot(net_output(idx).ROC_X_vote{j}, net_output(idx).ROC_Y_vote{j}); 
    end
     plot([0, 1], [0, 1], 'k--');
    xlabel('False positive rate')
    ylabel('True positive rate');
     legend('All Colors', 'Data 1', 'Data 2', 'Data 3', 'Data 4', 'Data 5',...
        'Location', 'southeast')
   
    
    figure()
    title('ROC ensemble')
    hold on;
    plot(ens_output(idx).ROC_X, ens_output(idx).ROC_Y);
    plot([0, 1], [0, 1], 'k--');
    xlabel('False positive rate')
    ylabel('True positive rate');
    
    for j = 1:length(net_output(idx).trainAcc)
        net_trainAcc = net_output(idx).trainAcc{j};
        net_validAcc = net_output(idx).validAcc{j};
        net_voteAcc = net_output(idx).voteAcc{j};

        figure()
        hold on;
        plot(net_trainAcc)
        plot(net_validAcc)
        plot(net_voteAcc)
        plot([0, length(net_trainAcc)], [0.5, 0.5], 'k--')
        
        title(sprintf('Image Package %d', [j]))
        legend('Train', 'Valid', 'Vote')
        xlabel('Epoch')
        ylabel('Accuracy')
        axis([0 length(net_trainAcc) 0 1])
    end

end

