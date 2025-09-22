function pred = test_lda(X, param)
    % X is a vector of data points
    % param is the output of fit_lda
    c0 = param(1,1);
    d0 = param(1,2);
    c1 = param(2,1);
    d1 = param(2,2);
    N = length(X);
    pred = zeros(N,1);

    for i = 1:N
        score0 = c0 * X(i) + d0;
        score1 = c1 * X(i) + d1;
        if score0 > score1
            pred(i) = 0;
        else
            pred(i) = 1;
        end
    end
end