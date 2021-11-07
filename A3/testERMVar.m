
figure;
for i=0.05:0.1:10
    plot(i,calc_ermperror(i), '.r');
    hold on;
end
title('min p-error w/ true PDF on tetrahedron 4class dataset');
xlabel('variance per class');
ylabel('p-error');
hold off;

disp(calc_ermperror(1));

function perror = calc_ermperror(var)
    covall = [var 0 0; 0 var 0; 0 0 var];
    m0  = [1; 1; 1];
    m1  = [-1; -1; 1];
    m2  = [-1; 1; -1];
    m3  = [1; -1; -1];
    
    samples = 10000;

    u = rand(1, samples);
    distr1 = length(find(u<=0.25));
    distr2 = length(find(u<=0.5))-distr1;
    distr3 = length(find(u<=0.75))-distr2-distr1;
    distr4 = length(find(u<= 1))-distr3-distr2-distr1;

    data1 = mvnrnd(m0,  covall, distr1);
    data2 = mvnrnd(m1,  covall, distr2);
    data3 = mvnrnd(m2, covall, distr3);
    data4 = mvnrnd(m3, covall, distr4);
    
    corr = 0;
    
    for i=1:distr1
        sample = data1(i, :).';
        if (epdf(sample,var)==1)
           corr = corr + 1;
        end
    end
    
    for i=1:distr2
        sample = data2(i, :).';
        if (epdf(sample,var)==2)
           corr = corr + 1;
        end
    end
    
    for i=1:distr3
        sample = data3(i, :).';
        if (epdf(sample,var)==3)
           corr = corr + 1;
        end
    end
    
    for i=1:distr4
        sample = data4(i, :).';
        if (epdf(sample,var)==4)
           corr = corr + 1;
        end
    end
    perror = (samples-corr)/samples;
end

function [choice] = epdf(x,var)
    covall = [var 0 0; 0 var 0; 0 0 var];
    m0  = [1; 1; 1];
    m1  = [-1; -1; 1];
    m2  = [-1; 1; -1];
    m3  = [1; -1; -1];
    
    pdf0 = evalGaussian(x,m0,covall);
    pdf1 = evalGaussian(x,m1,covall);
    pdf2 = evalGaussian(x,m2,covall);
    pdf3 = evalGaussian(x,m3,covall);
    [~, choice] = max([pdf0 pdf1 pdf2 pdf3]);
end

