clear all;

% utilize corners of a tetrahedron for our gaussian (so they are all
% equidistant)


m1  = [1; 1; 1];

m2  = [-1; -1; 1];

m31 = [-1; 1; -1];

m32 = [1; -1; -1];

samples = 10000;

u = rand(1, samples);
distr1 = length(find(u<=0.3));
distr2 = length(find(u<=0.6))-distr1;
distr3 = length(find(u < 0.8))-distr2-distr1;
distr4 = length(u)-distr3-distr2-distr1;


var_desired = (norm(m1-m2)/2)^2;



covall = [var_desired 0 0; 0 var_desired 0; 0 0 var_desired];

data1 = mvnrnd(m1, covall, distr1);
data2 = mvnrnd(m2, covall, distr2);
data3 = mvnrnd(m31, covall, distr3);
data4 = mvnrnd(m32, covall, distr4);

disp([distr1 distr2 distr3 distr4]);

figure;
grid on;

plot3(data1(:, 1),data1(:, 2),data1(:, 3), 'bo'); hold on;
plot3(data2(:, 1),data2(:, 2),data2(:, 3), 'r*'); hold on;
plot3(data3(:, 1),data3(:, 2),data3(:, 3), 'g+'); hold on;
plot3(data4(:, 1),data4(:, 2),data4(:, 3), 'm.'); hold on;

x = cat(1,cat(1, cat(1, data1, data2), data3), data4).';
y = cat(1,cat(1, cat(1, ones([distr1, 1]), 2*ones([distr2,1])), 3*ones([distr3, 1])), 3*ones([distr4, 1])).';

%save('q2data.mat', 'x','y', 'm1', 'm2', 'm31', 'm32', 'covall');
%save('q2pdf.mat', 'm1', 'm2', 'm31', 'm32', 'covall');