function result = fit_lasso_regression(X, y, lambdaGrid, kFold)
%FIT_LASSO_REGRESSION 使用K折交叉验证调参的Lasso回归。
    if nargin < 3 || isempty(lambdaGrid)
        lambdaGrid = [0.001, 0.01, 0.1, 1, 10, 100];
    end
    if nargin < 4 || isempty(kFold)
        kFold = 5;
    end

    % lasso 函数要求 Lambda 按降序排列
    lambdaGrid = sort(lambdaGrid(:)', 'descend');

    [B, fitInfo] = lasso(X, y, ...
        'Lambda', lambdaGrid, ...
        'Standardize', false, ...
        'CV', kFold, ...
        'RelTol', 1e-4, ...
        'MaxIter', 1e5);

    bestIdx = fitInfo.IndexMinMSE;
    model = struct('intercept', fitInfo.Intercept(bestIdx), 'beta', B(:, bestIdx));

    result = struct();
    result.model = model;
    result.lambda = fitInfo.Lambda(bestIdx);
    result.cvMSE = fitInfo.MSE;
    result.lambdaGrid = fitInfo.Lambda;
    result.coefPath = B;
end
