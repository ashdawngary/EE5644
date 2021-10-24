% Linear models (max deg 1)
clear all;
load("train10000.mat");
load("validate.mat");


x = train_10000x;
y = train_10000y;

loss_f = @(param)(bin_crossloss(x, y, param));
opt_opt = optimset('MaxIter', 2000,'MaxFunEvals', 1000000, 'Display', 'iter', 'TolFun',1e-6);

%initial_vec = [mean(y); 1e-2*(rand()-0.5); 1e-2*(rand()-0.5); 1e-2*(rand()-0.5); 1e-2*(rand()-0.5); 1e-2*(rand()-0.5)];
initial_vec = [-0.8563;1.2111;0.0338;-0.2762;0.3402;-0.1417];
opt_theta = fminsearch(loss_f, initial_vec, opt_opt);

figure;
validate_scatter = gscatter(validatex(1,:).', validatex(2,:).',validatey);
cont_f = @(x,y) arrayfun(@(x0,y0) h([x0;y0],opt_theta),x,y);
hold on;
g = gca;
fc = fcontour(cont_f,[g.XLim g.YLim]);
fc.LineStyle='--';
fc.LevelList = 1:-0.1:0;
colorbar
legend(validate_scatter,'Class label 0', 'Class label 1')

hold off;

mthresh = -100;
merror = 100;
figure;

for gamma=-0.05:0.001:1.05
   [T, F, perror] = compute_roc_point(opt_theta, validatex,validatey,gamma);
   plot(gamma, perror, '.r');
   if(perror < merror)
      merror = perror;
      mthresh = gamma;
   end
   hold on;
end

disp([mthresh merror]);

[T, F, perror] = compute_roc_point(opt_theta, validatex,validatey,mthresh);
plot(mthresh, perror, '+g');
   
   
hold off;

figure;
for gamma=-0.05:0.001:1.05
   [T, F, perror] = compute_roc_point(opt_theta, validatex,validatey,gamma);
   plot(F, T, '.r');
   hold on;
end

[T, F, perror] = compute_roc_point(opt_theta, validatex,validatey,mthresh);
plot(F, T, '+g');
hold off;

    



function [TP, FP, perror] = compute_roc_point(theta,x,y, gamma)
    [dim,samp] = size(x);
    res = zeros(size(y));
    
    for i=1:samp
        res(i) = h(x(:,i), theta);
    end
    
    y = logical(y);
    count1 = sum(y);
    count0 = sum(not(y));
    
    desc =  res >= gamma;
    
    TP = sum(logical(y) & desc)/count1;
    FP = sum(not(logical(y)) & desc)/count0;
    perror = 1 - (sum(desc == y)/samp);
end




function vec = b(x)
    x1 = x(1);
    x2 = x(2);
    vec = [1 ;x1; x2; x1*x1; x1*x2; x2*x2];
end

function p = h(x, theta)
    presig = -theta.'*b(x);
    p = 1/(1 + exp(presig));
end

function loss = bin_crossloss(x, labels, theta)
    csu = 0;
    [~,samp] = size(labels);
    for i=1:samp
        hvalue = h(x(:, i), theta);
        csu =  csu + (1-labels(i))*(log(1-hvalue)) + labels(i)*(log(hvalue));
    end
    
    loss = -1/samp * csu;
end

