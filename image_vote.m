function [confMat_vote, score_vote, correct, patients] = ...
    image_vote(data_set, score)
    % Function for voting the different crops into a single score per
    % patient

    global data_sheet
    tbl = data_sheet(:, 2:4);

    predict = [];
    correct = [];
    patients = [];
    score_vote = [];
    
    idx_mat = [];
    
    % Loop through all possible patient paths, for each paths find all crops
    % belonging to that path and vote.
    for i=1:9

        glas_s=['Glas', num2str(i),'_'];
        for k=1:120
            pat_s=[glas_s, num2str(k),'_'];

            ind=(strfind(data_set.Files,pat_s));
            ix=cellfun('isempty', ind);
            ix=ix==0;
            idx=find(ix==1);
            if (size(idx,1)==0)
                continue
            else
                
                idx_mat = [idx_mat, idx'];

                correct = [correct,  grp2idx(data_set.Labels(idx(1))) - 1];
                if (mean(score(idx)) > 0.5)
                    predict = [predict, 1];
                else
                    predict = [predict, 0];
                end
                score_vote = [score_vote, mean(score(idx))];
                            
                glas = glas_s(1:end-1);
                id = k;
                
                ind = strfind(tbl(:, 1), glas);
                idx = cellfun('isempty', ind) == 0;
                
                tmp_tbl = tbl(idx, :);
                idx = cell2mat(tmp_tbl(:, 2)) == k;
                
                core = tmp_tbl(idx, :);
                
                patients = [patients, core{3}];
                
            end
        end
    end

    confMat = confusionmat(correct, predict);
    confMat_vote = confMat./repmat(sum(confMat, 2), 1, ...
        size(confMat, 2));
  
end