function lasso_reg()
%LASSO_REG Quick demo highlighting L1 vs L2 behaviour on synthetic data.
%
%   Running this function spawns illustrative plots showing how Lasso can
%   drive coefficients to zero compared to ridge regression and OLS.

    rng(42);
    n = 50; p = 10;
    X = randn(n, p);
    true_beta = [2, -1.5, 1, zeros(1, p-3)]';
    y = X * true_beta + 0.5 * randn(n, 1);

    fprintf('真实情况：只有前3个特征有影响，后7个是噪声特征\n');
    fprintf('真实非零系数个数：%d\n', sum(true_beta ~= 0));

    figure('Position', [100, 100, 1000, 500]);
    subplot(1,2,1);
    beta_ols = linear_regression(X, y);
    beta_ridge = ridge_regression(X, y, 1);
    beta_lasso = lasso_regression(X, y, 1);

    stem(1:p, [beta_ols.beta, beta_ridge.beta, beta_lasso.beta]);
    xlabel('Feature index');
    ylabel('Coefficient value');
    legend('OLS', 'Ridge', 'Lasso');
    title('Coefficient comparison for λ = 1');
    grid on;

    subplot(1,2,2);
    lambda_range = logspace(-2, 2, 50);
    ridge_path = zeros(p, numel(lambda_range));
    for i = 1:numel(lambda_range)
        model = ridge_regression(X, y, lambda_range(i));
        ridge_path(:, i) = model.beta;
    end
    [lasso_path, fit_info] = lasso(X, y, 'Lambda', lambda_range, 'Standardize', false);

    semilogx(lambda_range, ridge_path', 'LineWidth', 1.2);
    hold on;
    semilogx(lambda_range, lasso_path', '--', 'LineWidth', 1.2);
    xlabel('\lambda');
    ylabel('Coefficient value');
    title('Ridge vs Lasso Coefficient Paths');
    grid on;
    legend_strings = arrayfun(@(i) sprintf('Feature %d', i), 1:p, 'UniformOutput', false);
    legend(legend_strings, 'Location', 'bestoutside');
    hold off;

    fprintf('Lasso默认选取的λ: %.4f\n', fit_info.LambdaMinMSE);
end
