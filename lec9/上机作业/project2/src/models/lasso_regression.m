function model = lasso_regression(X, y, lambda)
%LASSO_REGRESSION Wrapper around MATLAB's lasso function.
%
%   INPUTS:
%       X (matrix): training features.
%       y (vector): targets.
%       lambda (scalar): L1 penalty. If empty, uses lasso's default path.
%
%   OUTPUTS:
%       model struct with:
%           beta        - coefficient vector
%           intercept   - intercept term
%           predict     - prediction handle
%           lambda      - lambda used
%           fit_info    - struct from lasso when available

    arguments
        X double
        y double
        lambda double {mustBeNonnegative} = []
    end

    X = ensure_matrix(X);
    y = ensure_column_vector(y);

    opts = {'Standardize', false};
    if isempty(lambda)
        [beta_path, fit_info] = lasso(X, y, opts{:});
        idx = fit_info.Index1SE;
        beta = beta_path(:, idx);
        intercept = fit_info.Intercept(idx);
        lambda_used = fit_info.Lambda(idx);
    else
        [beta_path, fit_info] = lasso(X, y, 'Lambda', lambda, opts{:});
        beta = beta_path;
        intercept = fit_info.Intercept;
        lambda_used = lambda;
    end

    model = struct();
    model.beta = beta;
    model.intercept = intercept;
    model.predict = @(Xnew) intercept + Xnew * beta;
    model.lambda = lambda_used;
    model.fit_info = fit_info;
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
