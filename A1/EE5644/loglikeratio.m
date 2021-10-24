function logratios = loglikeratio(x)
    logpdf1 = log(evalPDF1(x));
    logpdf2 = log(evalPDF0(x));
    logratios =  logpdf1-logpdf2;
end

function likelihoods = evalPDF0(x)
    eval1 = 0.5*evalGaussian(x, [3;0], [2,0;0,1]);
    eval2 = 0.5*evalGaussian(x, [0;3], [1,0;0,2]);
    likelihoods = eval1 + eval2;
end
function likelihoods = evalPDF1(x)
    likelihoods = evalGaussian(x, [2;2], [1,0;0,1]);
end