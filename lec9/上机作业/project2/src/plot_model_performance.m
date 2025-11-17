function plot_model_performance(modelNames, mses, nonZeroCounts, savePath)
%PLOT_MODEL_PERFORMANCE 绘制三种模型的MSE与稀疏性对比。
    if nargin < 4
        savePath = '';
    end

    figure('Position', [100, 100, 1000, 400]);

    subplot(1, 2, 1);
    bar(mses, 'FaceColor', [0.2 0.4 0.8]);
    xticks(1:numel(modelNames));
    xticklabels(modelNames);
    ylabel('测试集 MSE');
    title('模型误差比较');
    grid on;

    subplot(1, 2, 2);
    bar(nonZeroCounts, 'FaceColor', [0.8 0.4 0.2]);
    xticks(1:numel(modelNames));
    xticklabels(modelNames);
    ylabel('非零系数个数');
    title('模型稀疏性对比');
    grid on;

    if ~isempty(savePath)
        exportgraphics(gcf, savePath, 'Resolution', 300);
    end
end
