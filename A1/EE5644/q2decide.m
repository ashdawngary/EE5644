function desc = q2decide(x, loss)
    [~, samp] = size(x);
    desc = zeros([1, samp]);
    risks = riskvec(x, loss);
    for ns = 1:samp
        classrisks = risks(:, ns);
        [~, cmin] = min(classrisks);
        desc(ns) = cmin; % cmin is argmin of classrisks.
    end

end
function risk = riskvec(x, loss)
    risk = loss*classprobs(x);
end
function p_l_given_x = classprobs(x)
    priors = [0.3 0 0; 0 0.3 0; 0 0 0.4];
    matr = priors* pdfall(x);
    [~,nsamp] = size(matr);
    p_l_given_x = zeros([3, nsamp]);
    
    for i=1:nsamp
        mcol = matr(:, i);
        p_l_given_x(:, i) = (1/sum(mcol))*mcol;
    end
    
    
    %p_l_given_x = (1/sum(matr))*matr; % normalized. or dividing by 1/p(x)
end

function p_x_given_l = pdfall(x)
    p_x_given_l = [pdf1(x); pdf2(x); pdf3(x)];
end

function p = pdf1(x)
    load('q2pdf.mat', 'm1', 'covall');
    p = evalGaussian(x, m1, covall);
end

function p = pdf2(x)
    load('q2pdf.mat', 'm2', 'covall');
    p = evalGaussian(x, m2, covall);
end


function p = pdf3(x)
    load('q2pdf.mat', 'm31', 'm32', 'covall');
    p1 = evalGaussian(x, m31, covall);
    p2 = evalGaussian(x, m32, covall);
    p = 0.5*p1+0.5*p2;
end