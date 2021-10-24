clear all;
m01 = [5; 0];
c01 = [4 0; 0 2];
m02 = [0; 4];
c02 = [1 0 ; 0 3];
m1 = [3;2];
c1 = [2 0; 0 2];

[train_100x, train_100y] = make_samples(100,m01, c01,m02,c02,m1,c1);
[train_1000x, train_1000y] = make_samples(1000,m01, c01,m02,c02,m1,c1);
[train_10000x, train_10000y] = make_samples(10000,m01, c01,m02,c02,m1,c1);
[validatex, validatey] = make_samples(20000,m01, c01,m02,c02,m1,c1);


%save("train100.mat", "train_100x", "train_100y");
%save("train1000.mat", "train_1000x", "train_1000y");
%save("train10000.mat", "train_10000x", "train_10000y");
%save("validate.mat", "validatex", "validatey");
%save("pdf.mat", "m01", "c01", "m02", "c02", "m1", "c1");





function [datax, datay] = make_samples(nsamples, m01, c01,m02,c02,m1,c1)
    %global m01 c01 m02 c02 m1 c1;
    split = rand(1, nsamples);
    
    count01 = length(find(split <= 0.3));
    count02 = length(find(split <= 0.6 & split >= 0.3));
    count1 = length(find(split <= 1 & split >= 0.6));
    
    data01 =  mvnrnd(m01, c01, count01);
    data02 =  mvnrnd(m02, c02, count02);
    data1 =  mvnrnd(m1, c1, count1);
    
    datax = cat(1, cat(1, data01, data02), data1).';
    datay = [zeros([1 count01+count02]) ones([1 count1])];

end