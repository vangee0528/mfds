function model = linear_regression(X, y)
%LINEAR_REGRESSION Closed-form ordinary least squares estimator.
%
%   INPUTS:
%       X (matrix): training features (observations x features).
%       y (vector): training targets.
%
%   OUTPUTS:
%       model (struct) containing:
%           beta      - estimated coefficients (features x 1)
%           intercept - scalar intercept term
%           predict   - function handle @(Xnew) -> yhat

    arguments
        X double
        y double
    end

    X = ensure_column_major(X);
    y = ensure_column_vector(y);

    X_aug = [ones(size(X, 1), 1), X];
    theta = X_aug \ y;

    intercept = theta(1);
    beta = theta(2:end);

    model = struct();
    model.beta = beta;
    model.intercept = intercept;
    model.predict = @(Xnew) intercept + Xnew * beta;
end

function X = ensure_column_major(X)
% Ensure the input is double and column-major for downstream ops.
    if ~isa(X, 'double')
        X = double(X);
    end
end

function vec = ensure_column_vector(vec)
% Coerce inputs into a column vector representation.
    if isrow(vec)
        vec = vec';
    end
    if ~isa(vec, 'double')
        vec = double(vec);
    end
end
