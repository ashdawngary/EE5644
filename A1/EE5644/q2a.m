clear all;

load("q2data.mat");
loss = ones([3, 3])-eye(3);

disp("using 0-1 loss:");
disp(loss);
desc = q2decide(x, loss);
cfm = make_confmatrix(desc, y, 3);

[~, nsamp] = size(x);


figure;
class_sym = ['.', '*', 'o'];
for samp =1:nsamp
   color = 'g'; % correct
   if desc(samp) ~= y(samp)
       color='r';
   end
   
   sym = class_sym(y(samp));
 
   marker = [color sym];
   plot3( x(1, samp), x(2, samp), x(3, samp), marker);    hold on;
end

hold off;

title('classification results');
disp('confusion matrix: ');
disp(cfm);
disp('accuracy: ');
disp(cfm(1,1)*0.3  + cfm(2,2)*0.3 + cfm(3,3)*0.4);
exp_loss_matrix = cfm.*loss;

loss_sums = sum(exp_loss_matrix);
expected_loss = sum(loss_sums.* [0.3 0.3 0.4]); % multiply by class priors.

disp('expected loss: ');
disp(expected_loss);