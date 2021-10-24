clear all, close all,

% Generate n-dimensional data vectors from 2 Gaussian pdfs
n = 2; 
N1 = 100; mu1 = -1*ones(n,1); A1 = 3*(rand(n,n)-0.5); %S1 = A*A';
N2 = 75; mu2 = 1*ones(n,1); A2 = 2*(rand(n,n)-0.5);
x1 = A1*randn(n,N1)+mu1*ones(1,N1);
x2 = A2*randn(n,N2)+mu2*ones(1,N2);

% Estimate mean vectors and covariance matrices from samples
mu1hat = mean(x1,2); S1hat = cov(x1');
mu2hat = mean(x2,2); S2hat = cov(x2');

% Calculate the between/within-class scatter matrices
Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

% Solve for the Fisher LDA projection vector (in w)
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

% Linearly project the data from both categories on to w
y1 = w'*x1;
y2 = w'*x2;

