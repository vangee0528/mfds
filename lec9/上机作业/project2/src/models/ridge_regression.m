function model = ridge_regression(X, y, lambda)
%RIDGE_REGRESSION Closed-form ridge regression with intercept exclusion.
%
%   INPUTS:
%       X (matrix): training features.
%       y (vector): training targets.
%       lambda (scalar): L2 regularization strength (> 0).
%
%   OUTPUTS:
%       model struct with fields beta, intercept, predict, lambda.

    arguments
        X double
        y double
        lambda double {mustBeNonnegative}
    end

    if lambda < 0
        error('ridge_regression:InvalidLambda', 'Lambda must be non-negative.');
    end

    X = ensure_matrix(X);
    y = ensure_column_vector(y);

    n_features = size(X, 2);
    X_aug = [ones(size(X, 1), 1), X];

    % Construct regularization matrix that skips the intercept term.
    reg = lambda * eye(n_features + 1);
    reg(1, 1) = 0;

    theta = (X_aug' * X_aug + reg) \ (X_aug' * y);
    intercept = theta(1);
    beta = theta(2:end);

    model = struct();
    model.beta = beta;
    model.intercept = intercept;
    model.predict = @(Xnew) intercept + Xnew * beta;
    model.lambda = lambda;
end

function X = ensure_matrix(X)
% Coerce to double matrix.
    if ~isa(X, 'double')
        X = double(X);
    end
end

function vec = ensure_column_vector(vec)
% Ensure column double vector.
    if isrow(vec)
        vec = vec';
    end
    if ~isa(vec, 'double')
        vec = double(vec);
    end
end
