clear all;
figure;
hold on;
load("q1data.mat")

for i=-20:0.1:20
   [TP, FP] = compute_roc_point(x,y,exp(i));
   plot(FP, TP, '.r');
   %disp([i round(TP, 3) round(FP, 3)]);
end

% gamma mAp is at P(L = 0)/P(L = 1)

mAp_gamma = sum(not(logical(y)))/sum(logical(y));

[TP, FP] = compute_roc_point(x, y, mAp_gamma);
plot(FP, TP, 'xb');
xlabel('False Positive (FP)');
ylabel('True Positive (TP)');
% ROC above


% threshold vs prob error below

figure;
hold on;
[~, samples] = size(x);

mthresh = -100;
merror = 1;

for i=-2.5:0.01:4
    desc = decide(x, exp(i));
    perror = 1- sum(desc == y)/samples;
    plot(i, perror, '.r');
    if(perror < merror)
       merror = perror;
       mthresh = exp(i);
    end
end


desc = decide(x, mAp_gamma);
perror = 1- sum(desc == y)/samples;
if(perror < merror)
   merror = perror;
   mthresh = mAp_gamma;
end
plot(log(mAp_gamma), perror, 'xb');
disp("minimum is at ");
disp(mthresh);
disp("mAp threshold is at");
disp(mAp_gamma);

xlabel('log(gamma)');
ylabel('p-error');
desc = decide(x, mthresh);
perror = 1- sum(desc == y)/samples;
plot(log(mthresh), perror, '+g');




