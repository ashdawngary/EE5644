function cfm  = make_confmatrix(y_pred, y_truth, nclass)
    cfm  = zeros([nclass, nclass]);
    for pred_class = 1:nclass
        for truth_class= 1:nclass
            common = sum((y_pred == pred_class) & (y_truth == truth_class));
            cfm(pred_class, truth_class) = common;
        end
    end
    for truth_class= 1:nclass
       cvec = cfm(:, truth_class);
       cvec = cvec * 1/sum(cvec);
       cfm(:, truth_class) = cvec;
    end
end