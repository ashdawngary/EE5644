clear all;
samples = 10000;

m01 = [3;0];
c01 = [2 0; 0 1];

m02 = [0;3];
c02 = [1 0; 0 2];

split = rand(1, 10000);

nsamples01 = length(find(split <= 0.325));
nsamples02 = length(find(split > 0.325 & split < 0.65));


m1 = [2;2];
c1 = [1 0 ; 0 1];
nsamples1 = length(find(split >= 0.65));

g01 = mvnrnd(m01, c01, nsamples01);
g02 = mvnrnd(m02, c02, nsamples02);
g1 = mvnrnd(m1, c1, nsamples1);
figure(1);
title('plot of individual gaussians')
subplot(3,1,1);plot(g01(:, 1), g01(:, 2), 'xb'); 
subplot(3,1,2);plot(g02(:, 1), g02(:, 2), '.r');
subplot(3,1,3);plot(g1(:, 1), g1(:, 2), 'og');   
figure();
title('plot of all gaussians together')
plot(g01(:, 1), g01(:, 2), 'xb'); hold on; axis equal;
plot(g02(:, 1), g02(:, 2), '.b'); 
plot(g1(:, 1), g1(:, 2), 'og');



x = cat(1, cat(1, g01, g02), g1).';
y = [zeros([1, nsamples01+nsamples02]) ones([1 nsamples1])];

testmean = mean(g01);
testcov = cov(g01);

%save('q1data.mat', 'x','y', 'm01', 'm02', 'm1', 'c01', 'c02', 'c1');