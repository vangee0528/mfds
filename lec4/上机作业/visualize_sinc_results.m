function savedPaths = visualize_sinc_results(results, config)
%VISUALIZE_SINC_RESULTS 可视化自定义神经网络的二维 sinc 拟合结果
%
%   savedPaths = VISUALIZE_SINC_RESULTS(results, config) 使用
%   train_sinc_network 返回的结果结构体，对真值、预测值、误差以及
%   训练过程进行可视化，并将图片与关键指标保存至指定目录。

validate_visual_config(config);

predict = results.predict;
output_dir = config.output_dir;
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 生成可视化网格
[x_grid, y_grid] = meshgrid(config.domain_min:config.grid_step:config.domain_max);
r_grid = sqrt(x_grid.^2 + y_grid.^2);
true_grid = sinc(r_grid / pi);

grid_inputs = [x_grid(:)'; y_grid(:)'];
pred_grid = reshape(predict(grid_inputs), size(x_grid));
abs_err_grid = abs(pred_grid - true_grid);

savedPaths = struct();

% 真值曲面
fig_true = figure('Visible', 'off');
surf(x_grid, y_grid, true_grid, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('二维 Sinc 函数真值');
colorbar; shading interp;
true_path = fullfile(output_dir, 'sinc_true_surface.png');
exportgraphics(fig_true, true_path, 'Resolution', 300);
close(fig_true);
savedPaths.trueSurface = true_path;

% 预测曲面
fig_pred = figure('Visible', 'off');
surf(x_grid, y_grid, pred_grid, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('神经网络预测的二维 Sinc 函数');
colorbar; shading interp;
pred_path = fullfile(output_dir, 'sinc_pred_surface.png');
exportgraphics(fig_pred, pred_path, 'Resolution', 300);
close(fig_pred);
savedPaths.predSurface = pred_path;

% 绝对误差
fig_err = figure('Visible', 'off');
surf(x_grid, y_grid, abs_err_grid, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('|误差|');
title('预测绝对误差分布');
colorbar; shading interp;
err_path = fullfile(output_dir, 'sinc_abs_error_surface.png');
exportgraphics(fig_err, err_path, 'Resolution', 300);
close(fig_err);
savedPaths.errorSurface = err_path;

% 训练/验证损失曲线
fig_perf = figure('Visible', 'off');
epochs = (1:numel(results.history.trainLoss))';
semilogy(epochs, results.history.trainLoss, 'b-', 'LineWidth', 1.8); hold on;
semilogy(epochs, results.history.valLoss, 'g--', 'LineWidth', 1.8);
xlabel('迭代轮数'); ylabel('均方误差 (log10)');
legend('训练损失', '验证损失', 'Location', 'northeast');
title('训练/验证性能曲线'); grid on;
perf_path = fullfile(output_dir, 'training_performance.png');
exportgraphics(fig_perf, perf_path, 'Resolution', 300);
close(fig_perf);
savedPaths.trainingPerformance = perf_path;

% 保存结果摘要
metrics_path = fullfile(output_dir, 'sinc_nn_results.txt');
fid = fopen(metrics_path, 'w');
fprintf(fid, '自定义前馈网络二维 sinc 拟合实验结果\n');
fprintf(fid, '------------------------------------------\n');
fprintf(fid, '样本总数: %d\n', results.counts.total);
fprintf(fid, '训练/验证/测试: %d / %d / %d\n', ...
        results.counts.train, results.counts.val, results.counts.test);
fprintf(fid, '网络结构: [%s]\n', strjoin(string(config.hidden_sizes), ', '));
fprintf(fid, '训练轮数: %d\n', config.max_epochs);
fprintf(fid, '学习率: %.4f, 批大小: %d\n', config.learning_rate, config.batch_size);
fprintf(fid, '最终训练 MSE: %.6e\n', results.history.trainLoss(end));
fprintf(fid, '最终验证 MSE: %.6e\n', results.history.valLoss(end));
fprintf(fid, '测试集平均相对误差: %.6e\n', results.testError);
fclose(fid);
savedPaths.metrics = metrics_path;

end

function validate_visual_config(config)
required = {"domain_min", "domain_max", "grid_step", "output_dir", ...
            "hidden_sizes", "max_epochs", "learning_rate", "batch_size"};
for k = 1:numel(required)
    if ~isfield(config, required{k})
        error('visualize_sinc_results:MissingConfig', ...
              '配置结构缺少字段 "%s"。', required{k});
    end
end
end
