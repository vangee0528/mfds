function plot_coeff_paths(ridgeLambda, ridgePath, lassoLambda, lassoPath, featureNames, savePath)
%PLOT_COEFF_PATHS 绘制岭回归与Lasso的系数路径。
    if nargin < 6
        savePath = '';
    end

    figure('Position', [100, 100, 1200, 500]);

    [ridgeLambdaSorted, ridgeIdx] = sort(ridgeLambda(:)', 'ascend');
    ridgePath = ridgePath(:, ridgeIdx);

    subplot(1, 2, 1);
    semilogx(ridgeLambdaSorted, ridgePath', 'LineWidth', 1.5);
    title('Ridge 系数路径');
    xlabel('\lambda'); ylabel('系数值');
    legend(featureNames, 'Location', 'bestoutside');
    grid on;

    [lassoLambdaSorted, lassoIdx] = sort(lassoLambda(:)', 'ascend');
    lassoPath = lassoPath(:, lassoIdx);

    subplot(1, 2, 2);
    semilogx(lassoLambdaSorted, lassoPath', 'LineWidth', 1.5);
    title('Lasso 系数路径');
    xlabel('\lambda'); ylabel('系数值');
    legend(featureNames, 'Location', 'bestoutside');
    grid on;

    if ~isempty(savePath)
        exportgraphics(gcf, savePath, 'Resolution', 300);
    end
end
