function param = fit_lda(data)
    X = data(:,1);
    Y = data(:,2);

    mu0 = mean(X(Y==0));
    mu1 = mean(X(Y==1));
    sigma_squared = sum((X-mean(X)).^2) / (length(X)-1);

    phi = sum(Y) / length(Y);
    c0 = mu0/sigma_squared;
    d0 = mu1/sigma_squared;
    d0 = log(1-phi) - c0 / 2
    c1 = mu1/sigma_squared;

    d1 = log(phi) - c1 / 2;
    param = [c0, d0; c1, d1];

end