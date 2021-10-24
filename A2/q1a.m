clear all;
load("validate.mat")





opt_gamma = 0.6/0.4;

figure;
hold on;
[~, samples] = size(validatex);

mthresh = -100;
merror = 1;

for i=-2.5:0.01:4
    desc = decide(validatex, exp(i));
    perror = 1- sum(desc == validatey)/samples;
    plot(i, perror, '.r');
    if(perror < merror)
       merror = perror;
       mthresh = exp(i);
    end
end


desc = decide(validatex, opt_gamma);
perror = 1- sum(desc == validatey)/samples;
if(perror < merror)
   merror = perror;
   mthresh = opt_gamma;
end
plot(log(opt_gamma), perror, 'xb');
disp("minimum is at ");
disp(mthresh);
disp("p-error opt threshold is at");
disp(opt_gamma);

xlabel('log(gamma)');
ylabel('p-error');
desc = decide(validatex, mthresh);
perror = 1- sum(desc == validatey)/samples;
plot(log(mthresh), perror, '+g');



figure;
hold on;
for i=-10:0.05:10
   [TP, FP] = compute_roc_point(validatex,validatey,exp(i));
   plot(FP, TP, '.r'); hold on;
   %disp([i round(TP, 3) round(FP, 3)]);
end

[TP, FP] = compute_roc_point(validatex,validatey,mthresh);
plot(FP, TP, '+b'); hold on;
 
hold off;

figure;
q = gscatter(validatex(1, :), validatex(2, :), validatey);
hold on;
llpdf = @(x,y) arrayfun(@(x0,y0) loglikeratio([x0;y0]),x,y);
g = gca;
fcontour(llpdf,[g.XLim g.YLim])
title('{\bf ERM Classifier descision boundary based on true pdf for validation set}')
legend(q,'Class label 0', 'class label 1')
hold off







function [TP, FP] = compute_roc_point(x, y, gamma)
    y = logical(y);
    count1 = sum(y);
    count0 = sum(not(y));
    
    desc =  decide(x, gamma);
    
    TP = sum(logical(y) & desc)/count1;
    FP = sum(not(logical(y)) & desc)/count0;
end



% decide for 1a/b log likelihood test.
function desc = decide(x, gamma)
    desc = loglikeratio(x) >= log(gamma);
end


function logratios = loglikeratio(x)
    logpdf1 = log(evalPDF1(x));
    logpdf2 = log(evalPDF0(x));
    logratios =  logpdf1-logpdf2;
end

function likelihoods = evalPDF0(x)
    eval1 = 0.5*evalGaussian(x, [5; 0], [4 0; 0 2]);
    eval2 = 0.5*evalGaussian(x, [0; 4], [1 0 ; 0 3]);
    likelihoods = eval1 + eval2;
end
function likelihoods = evalPDF1(x)
    likelihoods = evalGaussian(x, [3;2], [2 0; 0 2]);
end