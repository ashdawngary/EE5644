clear all;

load("q1data.mat");

% filp class labels
%y = not(logical(y));

[m0,cov0] = estimate_params(x(:, y==0));
[m1,cov1] = estimate_params(x(:, y==1));



mudiff = m0-m1;
sb = mudiff*mudiff.';
sw = cov0 + cov1;
tosolve = inv(sw)*sb;

[eV, eD] = eig(tosolve);

[~,ind] = sort(diag(eD),'descend');
w = eV(:,ind(1)); % vector with largegmst evalue

scores = w'*x;

minpe = 1;
mini = 100;

figure;
hold on;
xlabel('False Positive (FP)');
ylabel('True Positive (TP)');


for i=-10:0.05:10
   [T, F,pe] = roclda(scores, y, i);
   [TP, TN, FP, FN, PE] = conf(scores, y, i);
   disp([i TP TN FP FN PE]);
   
   %disp([i T F pe]) TP FP
   plot(FP, TP, 'xb');
   if pe < minpe
    minpe = pe;
    mini = i;
   end
end

[T, F,pe] = roclda(scores, y, mini);
plot(F, T, '+g');
   
figure;
hold on;
xlabel('tau');
ylabel('p-error');
for i=-10:0.05:10
   [~, ~,pe] = roclda(scores, y, i);
   
   plot(i,pe, 'xb');
   
end

[T, F,pe] = roclda(scores, y, mini);
plot(mini,pe, '+g');

disp("minimum p-error is at ");
disp(mini);
disp("with p-error of");
disp(minpe);

y1 = scores(logical(y)==0);
y2 = scores(logical(y)==1);

figure;
plot(y1(1, :), zeros(1, length(y1)), 'r*');hold on;
plot(y2(1, :), ones(1, length(y2)), 'bo');hold on;
plot(mini*ones(1, 40), 1/5 * (-20:19), 'g+');hold on;
ylim([-4,4]);
xlabel('projected value');
ylabel('class label');


figure;

title('wtx lda descision boundary'); hold on;
x1 = x(:, y==0);
x2 = x(:, y==1);
plot(x1(1, :), x1(2, :), 'r*'); hold on;
plot(x2(1, :), x2(2, :), 'b.'); hold on;


%w0x + w1y = c

y_left = (mini - (-10 * w(1)))/w(2);
y_right = (mini - (10 * w(1)))/w(2);

hold on
line([-10 10],[y_left y_right])
hold off





function [TP, FP, perror] = roclda(scores,y, tau)
    y = logical(y);
    
    count1 = sum(y);
    count0 = sum(not(y));
    
    desc = scores;
    desc(scores >= tau) = 1;
    desc(scores < tau) = 0;
    
    TP = sum(logical(y) & (desc == 1))/count1;
    FP = sum(not(logical(y)) & (desc == 1))/count0;
    
    perror = 1- sum(logical(desc) == logical(y))/length(scores);
end

function [TP, TN, FP, FN, PE] = conf(scores, y, tau)
    y = logical(y);
    
    count1 = sum(y);
    count0 = sum(not(y));
    
    desc = scores;
    desc(scores >= tau) = 1;
    desc(scores < tau) = 0;
    
    TP = sum(logical(y) & (desc == 1))/count1;
    FP = sum(not(logical(y)) & (desc == 1))/count0;
    TN = sum(not(logical(y)) & (desc == 0))/count0;
    FN = sum(logical(y) & (desc == 0))/count1;
    PE = 1- sum(logical(desc) == logical(y))/length(scores);
end
function [muhat,sigmahat] = estimate_params(data)
    muhat = mean(data.').';
    sigmahat = cov(data.').';
end

