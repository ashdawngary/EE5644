function [TP, FP] = compute_roc_point(x, y, gamma)
    y = logical(y);
    count1 = sum(y);
    count0 = sum(not(y));
    
    desc =  decide(x, gamma);
    
    TP = sum(logical(y) & desc)/count1;
    FP = sum(not(logical(y)) & desc)/count0;
end