clear all;
% estimate params from D10k train
load("train10000.mat");
load("validate.mat");

x = train_10000x;
y = train_10000y; 

x0 = x(:, (y==0));
%P = randperm(size(x0,2));
%x0 = x0(:,P);

y0 = y(:, (y==0));

x1 = x(:, (y==1));
y1 = y(:, (y==1));


options = statset('MaxIter',1000, 'Display', 'final', 'TolFun', 1e-6, 'TolTypeFun', 'abs');

global gmModel0 m1_hat c1_hat;

prior_0 = length(y0)/length(y);
gmModel0 = fitgmdist(x0.',2, 'Options',options, 'RegularizationValue', 0.01);
%disp(["Iterations: " gmModel0.NumIterations]);
% use the sample average
prior_1 = length(y1)/length(y);
m1_hat = mean(x1, 2);

% cov works too, but uses 1/n-1 instead of 1/n
%cov_hat = cov(x1.');

% use sample covariance method: sum = 0->n (xi-mu)(xi-mu)^T
[d, samp] = size(x1);
a = zeros([d d]);
for i=1:samp
   vec = x1(:, i);
   a = a + ( (vec-m1_hat)*(vec-m1_hat).');
end
c1_hat = 1/(samp) * a;



% gaussian estimation results

figure;
h = gscatter(x0(1,:).',x0(2,:).',y0);
hold on;
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gmModel0,[x0 y0]),x,y);
g = gca;
fcontour(gmPDF,[g.XLim g.YLim])
title('{\bf EM output on Class 0 with 2 gaussian components.}')
legend(h,'Class label 0')
hold off
figure;

h2 = gscatter(x1(1,:).',x1(2,:).',y1);
hold on;
gmPDF2 = @(x,y) arrayfun(@(x0,y0) evalGaussian([x0;y0],m1_hat, c1_hat ),x,y);
g2 = gca;
fcontour(gmPDF2,[g2.XLim g2.YLim]);

title('{\bf Estimations of Class 1 with 1 gaussian component.}')
legend(h2,'Class label 1')
hold off;
%gscatter(x(1,:).',x(2,:).',y);;





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

descbest = decide(validatex, exp(1.09));

disp("minimum is at ");
disp([mthresh merror]);
xlabel('log(gamma)');
ylabel('p-error');
desc = decide(validatex, mthresh);
perror = 1- sum(desc == validatey)/samples;
plot(log(mthresh), perror, '+g');



[~, validsize] = size(validatey);
desc = decide(validatex, mthresh);
perror = 1- sum(desc == validatey)/validsize;
disp("P error on validate:")
disp(perror);

figure;
hold on;
for i=-10:0.01:10
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
title('{\bf ERM Classifier descision boundary based on EM based estimation of D10k}')
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

function pdf0 = evalPDF0(x)
    global gmModel0;
    pdf0 = pdf(gmModel0, x.').'; 
end
function likelihoods = evalPDF1(x)
    global m1_hat c1_hat;
    likelihoods = evalGaussian(x,m1_hat , c1_hat);
end
