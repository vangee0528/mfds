function model = fit_linear_regression(X, y)
%FIT_LINEAR_REGRESSION 普通最小二乘回归（带截距）。
    X_aug = [ones(size(X, 1), 1), X];
    theta = X_aug \ y;
    model = struct('intercept', theta(1), 'beta', theta(2:end));
end
