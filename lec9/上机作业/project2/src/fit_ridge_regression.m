function result = fit_ridge_regression(X, y, lambdaGrid, kFold)
%FIT_RIDGE_REGRESSION 使用K折交叉验证调参的岭回归。
    if nargin < 3 || isempty(lambdaGrid)
        lambdaGrid = [0.001, 0.01, 0.1, 1, 10, 100];
    end
    if nargin < 4 || isempty(kFold)
        kFold = 5;
    end

    nSamples = size(X, 1);
    cvIndices = crossvalind('Kfold', nSamples, kFold);
    meanMSE = zeros(numel(lambdaGrid), 1);

    for li = 1:numel(lambdaGrid)
        lambda = lambdaGrid(li);
        foldMSE = zeros(kFold, 1);
        for fold = 1:kFold
            testMask = cvIndices == fold;
            trainMask = ~testMask;
            model = solve_ridge(X(trainMask, :), y(trainMask), lambda);
            y_pred = predict_regression(model, X(testMask, :));
            foldMSE(fold) = compute_mse(y(testMask), y_pred);
        end
        meanMSE(li) = mean(foldMSE);
    end

    [~, bestIdx] = min(meanMSE);
    bestLambda = lambdaGrid(bestIdx);
    bestModel = solve_ridge(X, y, bestLambda);

    coefPath = zeros(size(X, 2), numel(lambdaGrid));
    for li = 1:numel(lambdaGrid)
        model = solve_ridge(X, y, lambdaGrid(li));
        coefPath(:, li) = model.beta;
    end

    result = struct();
    result.model = bestModel;
    result.lambda = bestLambda;
    result.cvMSE = meanMSE;
    result.lambdaGrid = lambdaGrid(:)';
    result.coefPath = coefPath;
end

function model = solve_ridge(X, y, lambda)
    X_aug = [ones(size(X, 1), 1), X];
    nFeatures = size(X_aug, 2);
    penalty = eye(nFeatures);
    penalty(1, 1) = 0; % 截距不正则化

    theta = (X_aug' * X_aug + lambda * penalty) \ (X_aug' * y);
    model = struct('intercept', theta(1), 'beta', theta(2:end));
end
