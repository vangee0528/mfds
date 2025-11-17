function metrics = evaluate_model(y_true, y_pred, beta, params)
%EVALUATE_MODEL Compute regression metrics and sparsity indicators.
%
%   INPUTS:
%       y_true (vector) : ground truth targets.
%       y_pred (vector) : predicted targets.
%       beta  (vector)  : model coefficients used for sparsity analysis.
%       params (struct): configuration struct (expects coeff_threshold).
%
%   OUTPUTS:
%       metrics (struct) with fields:
%           mse              - mean squared error
%           mae              - mean absolute error
%           r2               - coefficient of determination
%           nonzero_features - count of coefficients above threshold
%           sparsity_ratio   - proportion of features retained

    arguments
        y_true double
        y_pred double
        beta double
        params struct
    end

    residuals = y_true - y_pred;
    mse = mean(residuals .^ 2);
    mae = mean(abs(residuals));

    ss_tot = sum((y_true - mean(y_true)).^2);
    ss_res = sum(residuals.^2);
    r2 = 1 - ss_res / ss_tot;

    threshold = params.coeff_threshold;
    nonzero_mask = abs(beta) > threshold;
    nonzero_features = sum(nonzero_mask);
    sparsity_ratio = nonzero_features / numel(beta);

    metrics = struct();
    metrics.mse = mse;
    metrics.mae = mae;
    metrics.r2 = r2;
    metrics.nonzero_features = nonzero_features;
    metrics.sparsity_ratio = sparsity_ratio;
end
