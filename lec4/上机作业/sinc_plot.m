%% 二维 sinc 函数拟合实验（神经网络）
% 网络结构：输入层2 -> 隐藏层1(20) -> 隐藏层2(20) -> 输出层1
% 训练迭代：1000 次；初始权重、偏置均来自标准正态分布
% 数据采样：[-8, 8]^2 内均匀分布，避开原点半径 1e-6 邻域

clc; clear;

%% 参数设定
rng(2025);                 % 固定随机种子，便于复现
N_total = 8000;            % 样本数 (<= 10000)
domain_min = -8;
domain_max = 8;
min_radius = 1e-6;         % 避免 r~0 造成除零

hidden_sizes = [20 20];
max_epochs = 1000;

% 图像输出目录
output_dir = fullfile(pwd, 'figures');
if ~exist(output_dir, 'dir')
	mkdir(output_dir);
end

%% 生成训练/测试数据
num_features = 2;
inputs = zeros(num_features, N_total);
targets = zeros(1, N_total);

generated = 0;
while generated < N_total
	remain = N_total - generated;
	candidate = domain_min + (domain_max - domain_min) * rand(num_features, remain);
	r_candidate = sqrt(sum(candidate.^2, 1));
	mask = r_candidate >= min_radius;
	valid_pts = candidate(:, mask);
	num_valid = size(valid_pts, 2);
	if num_valid == 0
		continue;
	end
	inputs(:, generated + (1:num_valid)) = valid_pts;
	r_valid = sqrt(sum(valid_pts.^2, 1));
	targets(:, generated + (1:num_valid)) = sinc(r_valid / pi); % MATLAB sinc: sin(pi*x)/(pi*x)
	generated = generated + num_valid;
end

% 划分训练/验证/测试集 (70/15/15)
N_train = round(0.7 * N_total);
N_val = round(0.15 * N_total);
idx = randperm(N_total);
train_idx = idx(1:N_train);
val_idx = idx(N_train+1 : N_train+N_val);
test_idx = idx(N_train+N_val+1 : end);

train_inputs = inputs(:, train_idx);
train_targets = targets(:, train_idx);
val_inputs = inputs(:, val_idx);
val_targets = targets(:, val_idx);
test_inputs = inputs(:, test_idx);
test_targets = targets(:, test_idx);

%% 构建并初始化神经网络
net = fitnet(hidden_sizes, 'trainlm');
net.layers{end}.transferFcn = 'purelin'; % 输出层线性
net.trainParam.epochs = max_epochs;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = true;
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:size(train_inputs, 2);
net.divideParam.valInd = size(train_inputs, 2) + (1:size(val_inputs, 2));
net.divideParam.testInd = [];

% 重新组织输入以满足 divideind 的索引方式
train_val_inputs = [train_inputs, val_inputs];
train_val_targets = [train_targets, val_targets];

% 配置网络并加载标准正态初始权重
net = configure(net, train_val_inputs, train_val_targets);
for layer = 1:numel(net.IW)
	for conn = 1:numel(net.IW(layer, :))
		if ~isempty(net.IW{layer, conn})
			net.IW{layer, conn} = randn(size(net.IW{layer, conn}));
		end
	end
end
for layer = 1:numel(net.LW)
	for conn = 1:numel(net.LW(layer, :))
		if ~isempty(net.LW{layer, conn})
			net.LW{layer, conn} = randn(size(net.LW{layer, conn}));
		end
	end
end
for layer = 1:numel(net.b)
	if ~isempty(net.b{layer})
		net.b{layer} = randn(size(net.b{layer}));
	end
end

%% 训练
[net, tr] = train(net, train_val_inputs, train_val_targets);

%% 测试集评估
pred_test = net(test_inputs);
abs_denominator = max(abs(test_targets), 1e-8);
relative_errors = abs(pred_test - test_targets) ./ abs_denominator;
test_err = mean(relative_errors);

fprintf('神经网络在测试集上的平均相对误差：%.6f\n', test_err);

%% 生成网格用于可视化
[x_grid, y_grid] = meshgrid(domain_min:0.25:domain_max);
r_grid = sqrt(x_grid.^2 + y_grid.^2);
true_grid = sinc(r_grid / pi);

grid_inputs = [x_grid(:)'; y_grid(:)'];
pred_grid = net(grid_inputs);
pred_grid = reshape(pred_grid, size(x_grid));
abs_err_grid = abs(pred_grid - true_grid);

%% 可视化并保存图片
% 1. 真值曲面
fig_true = figure('Visible', 'off');
surf(x_grid, y_grid, true_grid, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('二维 Sinc 函数真值');
colorbar; shading interp;
exportgraphics(fig_true, fullfile(output_dir, 'sinc_true_surface.png'), 'Resolution', 300);

% 2. 预测曲面
fig_pred = figure('Visible', 'off');
surf(x_grid, y_grid, pred_grid, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('神经网络预测的二维 Sinc 函数');
colorbar; shading interp;
exportgraphics(fig_pred, fullfile(output_dir, 'sinc_pred_surface.png'), 'Resolution', 300);

% 3. 绝对误差热图
fig_err = figure('Visible', 'off');
surf(x_grid, y_grid, abs_err_grid, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('|误差|');
title('预测绝对误差分布');
colorbar; shading interp;
exportgraphics(fig_err, fullfile(output_dir, 'sinc_abs_error_surface.png'), 'Resolution', 300);

% 4. 训练表现曲线
fig_perf = figure('Visible', 'off');
plotperform(tr);
title('训练/验证性能曲线');
exportgraphics(fig_perf, fullfile(output_dir, 'training_performance.png'), 'Resolution', 300);

close([fig_true, fig_pred, fig_err, fig_perf]);

%% 将评估指标保存到文件，便于生成报告
results_txt = fullfile(output_dir, 'sinc_nn_results.txt');
fid = fopen(results_txt, 'w');
fprintf(fid, '神经网络二维 sinc 拟合实验结果\n');
fprintf(fid, '-----------------------------------\n');
fprintf(fid, '样本总数: %d\n', N_total);
fprintf(fid, '训练/验证/测试: %d / %d / %d\n', N_train, N_val, numel(test_idx));
fprintf(fid, '网络结构: [%s]\n', strjoin(string(hidden_sizes), ', '));
fprintf(fid, '训练迭代次数: %d\n', max_epochs);
fprintf(fid, '测试集平均相对误差: %.6f\n', test_err);
fprintf(fid, '训练性能 (最后一次): %.6f\n', tr.perf(end));
if ~isempty(tr.vperf)
	fprintf(fid, '验证性能 (最后一次): %.6f\n', tr.vperf(end));
end
fclose(fid);

disp('所有图像与结果均已保存至 figures 目录。');