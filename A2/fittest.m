c = 10000;
count01 = c;
count02 = c;
m01 = [5; 0];
c01 = [4 0; 0 2];
m02 = [0; 4];
c02 = [1 0 ; 0 3];

data01 =  mvnrnd(m01, c01, count01);
data02 =  mvnrnd(m02, c02, count02);

x0 = cat(1, data01, data02).';

options = statset('MaxIter',1000, 'Display', 'iter', 'TolFun', 1e-6, 'TolTypeFun', 'abs');
gmModel0 = fitgmdist(x0.',2, 'Options',options, 'RegularizationValue', 0.01);

figure;
h = gscatter(x0(1,:).',x0(2,:).');
hold on;
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gmModel0,[x0 y0]),x,y);
g = gca;
fcontour(gmPDF,[g.XLim g.YLim])
title('{\bf EM output on Class 0 with 2 gaussian components.}')
legend(h,'Class label 0')
hold off