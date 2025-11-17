%% 糖尿病数据回归分析
clear; clc; close all;

%% 1. 数据加载
fprintf('=== 糖尿病数据回归分析===\n');

fprintf('1. 开始数据加载\n');

if ~exist('data/diabetes.csv', 'file')
    error('数据文件 data/diabetes.csv 不存在，请检查文件路径');
end

try
    data = readtable('data/diabetes.csv');
    fprintf('   数据加载成功\n');
catch ME
    error('数据加载失败: %s', ME.message);
end

fprintf('   数据基本信息:\n');
fprintf('   +----------------------+---------------------------+\n');
fprintf('   | 指标                 | 数值                      |\n');
fprintf('   +----------------------+---------------------------+\n');
fprintf('   | 数据维度            | %3d 行 × %2d 列          |\n', size(data, 1), size(data, 2));
fprintf('   | 变量名              | %s |\n', strjoin(data.Properties.VariableNames, ', '));
fprintf('   +----------------------+---------------------------+\n');

X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

% 数据质量验证
[n_samples, n_features] = size(X);
feature_names = {'AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'};

% 检查数据完整性
if any(isnan(X(:))) || any(isnan(y))
    warning('数据中存在缺失值，可能需要处理');
    missing_X = sum(isnan(X(:)));
    missing_y = sum(isnan(y));
    fprintf('   警告: 发现 %d 个特征缺失值, %d 个目标变量缺失值\n', missing_X, missing_y);
end

fprintf('\n   数据统计摘要:\n');
fprintf('   +----------------------+---------------------------+\n');
fprintf('   | 指标                 | 数值                      |\n');
fprintf('   +----------------------+---------------------------+\n');
fprintf('   | 样本数              | %3d                        |\n', n_samples);
fprintf('   | 特征数              | %3d                        |\n', n_features);
fprintf('   | 目标均值            | %7.2f                    |\n', mean(y));
fprintf('   | 目标标准差          | %7.2f                    |\n', std(y));
fprintf('   | 目标范围            | [%.2f, %.2f]              |\n', min(y), max(y));
fprintf('   +----------------------+---------------------------+\n');

fprintf('\n   特征统计 (均值/标准差):\n');
fprintf('   +------+-----------+-----------+\n');
fprintf('   | 特征 | 均值      | 标准差    |\n');
fprintf('   +------+-----------+-----------+\n');
for i = 1:n_features
    fprintf('   | %-4s | %9.2f | %9.2f |\n', ...
        feature_names{i}, mean(X(:,i)), std(X(:,i)));
end
fprintf('   +------+-----------+-----------+\n');

%% 2. 数据预处理
fprintf('\n2. 数据预处理...\n');

rng(42, 'twister'); 
split_ratio = 0.7;
cv = cvpartition(n_samples, 'HoldOut', 1-split_ratio);
train_idx = training(cv);
test_idx = test(cv);

X_train = X(train_idx, :);
y_train = y(train_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);

n_train = sum(train_idx);
n_test = sum(test_idx);
fprintf('   数据分割完成:\n');
fprintf('   +--------+------------+------------+\n');
fprintf('   | 集合   | 样本数     | 占比       |\n');
fprintf('   +--------+------------+------------+\n');
fprintf('   | 训练集 | %6d     | %6.1f%%   |\n', n_train, n_train/n_samples*100);
fprintf('   | 测试集 | %6d     | %6.1f%%   |\n', n_test, n_test/n_samples*100);
fprintf('   +--------+------------+------------+\n');

fprintf('   数据标准化:\n');

% 特征标准化 - 使用训练集统计量
mu_X_train = mean(X_train);
sigma_X_train = std(X_train);

% 检查零方差特征
zero_var_features = find(sigma_X_train < 1e-10);
if ~isempty(zero_var_features)
    warning('发现 %d 个零方差特征: %s', length(zero_var_features), ...
        strjoin(feature_names(zero_var_features), ', '));
    sigma_X_train(zero_var_features) = 1;
end

X_train_std = (X_train - mu_X_train) ./ sigma_X_train;
X_test_std = (X_test - mu_X_train) ./ sigma_X_train;

fprintf('   +--------------------------+----------------------+\n');
fprintf('   | 特征均值范围            | [%.2f, %.2f]        |\n', min(mu_X_train), max(mu_X_train));
fprintf('   | 特征标准差范围          | [%.2f, %.2f]        |\n', min(sigma_X_train), max(sigma_X_train));
fprintf('   +--------------------------+----------------------+\n');

standardize_target = true; 
if standardize_target
    mu_y_train = mean(y_train);
    sigma_y_train = std(y_train);
    y_train_std = (y_train - mu_y_train) ./ sigma_y_train;
    y_test_std = (y_test - mu_y_train) ./ sigma_y_train;
    fprintf('   +--------------------------+----------------------+\n');
    fprintf('   | 目标均值 (训练集)       | %7.2f              |\n', mu_y_train);
    fprintf('   | 目标标准差 (训练集)     | %7.2f              |\n', sigma_y_train);
    fprintf('   +--------------------------+----------------------+\n');
else
    % 只中心化
    mu_y_train = mean(y_train);
    y_train_centered = y_train - mu_y_train;
    fprintf('   - 目标变量中心化完成 (均值=%.2f)\n', mu_y_train);
end

fprintf('\n   预处理质量检查:\n');
fprintf('   +--------------------------+----------------------+\n');
fprintf('   | 训练集均值范围          | [%.2f, %.2f]        |\n', min(mean(X_train_std)), max(mean(X_train_std)));
fprintf('   | 训练集标准差范围        | [%.2f, %.2f]        |\n', min(std(X_train_std)), max(std(X_train_std)));
fprintf('   | 测试集均值范围          | [%.2f, %.2f]        |\n', min(mean(X_test_std)), max(mean(X_test_std)));
fprintf('   +--------------------------+----------------------+\n');

preprocess_params.mu_X = mu_X_train;
preprocess_params.sigma_X = sigma_X_train;
preprocess_params.mu_y = mu_y_train;
if standardize_target
    preprocess_params.sigma_y = sigma_y_train;
end
preprocess_params.feature_names = feature_names;

fprintf('   预处理参数已保存\n');

%% 3. 普通线性回归
fprintf('\n3. 普通线性回归...\n');

% 添加截距项
X_train_with_intercept = [ones(n_train, 1), X_train_std];
X_test_with_intercept = [ones(n_test, 1), X_test_std];

% 使用解析解
beta_linear = (X_train_with_intercept' * X_train_with_intercept) \ (X_train_with_intercept' * y_train_std);

% 预测
y_pred_linear = X_test_with_intercept * beta_linear * sigma_y_train + mu_y_train;

% 计算MSE - 核心指标
mse_linear = mean((y_test - y_pred_linear).^2);

% 特征数量
n_nonzero_linear = sum(beta_linear(2:end) ~= 0);

fprintf('   测试集MSE: %.4f\n', mse_linear);
fprintf('   非零特征数: %d/%d\n', n_nonzero_linear, n_features);

% 保存结果用于后续比较
results.linear.mse = mse_linear;
results.linear.beta = beta_linear;
results.linear.n_nonzero = n_nonzero_linear;
results.linear.y_pred = y_pred_linear;

%% 4. 岭回归
fprintf('\n4. 岭回归...\n');

lambda_values = [0.001, 0.01, 0.1, 1, 10, 100];
mse_ridge_values = zeros(size(lambda_values));
n_nonzero_ridge_values = zeros(size(lambda_values));
beta_ridge_all = zeros(n_features + 1, length(lambda_values));

fprintf('   Lambda调优:\n');
for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    
    % 岭回归解析解 
    I = eye(n_features + 1);
    I(1,1) = 0;
    
    beta_ridge = (X_train_with_intercept' * X_train_with_intercept + lambda * I) \ ...
                 (X_train_with_intercept' * y_train_std);
    beta_ridge_all(:, i) = beta_ridge;
    
    % 预测
    y_pred_ridge = X_test_with_intercept * beta_ridge * sigma_y_train + mu_y_train;
    mse_ridge_values(i) = mean((y_test - y_pred_ridge).^2);
    n_nonzero_ridge_values(i) = sum(beta_ridge(2:end) ~= 0);
    
    fprintf('   lambda=%.3f, MSE=%.4f, 非零特征: %d/%d\n', ...
        lambda, mse_ridge_values(i), n_nonzero_ridge_values(i), n_features);
end

% 选择最优岭回归模型
[best_ridge_mse, best_ridge_idx] = min(mse_ridge_values);
best_lambda_ridge = lambda_values(best_ridge_idx);
best_beta_ridge = beta_ridge_all(:, best_ridge_idx);
best_n_nonzero_ridge = n_nonzero_ridge_values(best_ridge_idx);

fprintf('   最优模型: lambda=%.3f\n', best_lambda_ridge);
fprintf('   测试集MSE: %.4f\n', best_ridge_mse);
fprintf('   非零特征数: %d/%d\n', best_n_nonzero_ridge, n_features);

% 保存结果
results.ridge.mse = best_ridge_mse;
results.ridge.beta = best_beta_ridge;
results.ridge.n_nonzero = best_n_nonzero_ridge;
results.ridge.beta_all = beta_ridge_all;
results.ridge.lambda_values = lambda_values;

%% 5. Lasso回归
fprintf('\n5. Lasso回归...\n');

% 使用相同的lambda范围
[B_lasso, FitInfo] = lasso(X_train_std, y_train_std, ...
    'Lambda', lambda_values, ...
    'Standardize', false);

mse_lasso_values = zeros(size(lambda_values));
n_nonzero_lasso_values = zeros(size(lambda_values));

fprintf('   Lambda调优:\n');
for i = 1:length(lambda_values)
    % 预测
    y_pred_lasso_std = X_test_std * B_lasso(:, i) + FitInfo.Intercept(i);
    y_pred_lasso = y_pred_lasso_std * sigma_y_train + mu_y_train;
    
    mse_lasso_values(i) = mean((y_test - y_pred_lasso).^2);
    n_nonzero_lasso_values(i) = sum(B_lasso(:, i) ~= 0);
    
    fprintf('   lambda=%.3f, MSE=%.4f, 非零特征: %d/%d\n', ...
        lambda_values(i), mse_lasso_values(i), n_nonzero_lasso_values(i), n_features);
end

% 选择最优Lasso模型（基于最小MSE）
[best_lasso_mse, best_lasso_idx] = min(mse_lasso_values);
best_lambda_lasso = lambda_values(best_lasso_idx);
best_beta_lasso = B_lasso(:, best_lasso_idx);
best_intercept_lasso = FitInfo.Intercept(best_lasso_idx);
best_n_nonzero_lasso = n_nonzero_lasso_values(best_lasso_idx);

fprintf('   最优模型: lambda=%.3f\n', best_lambda_lasso);
fprintf('   测试集MSE: %.4f\n', best_lasso_mse);
fprintf('   非零特征数: %d/%d\n', best_n_nonzero_lasso, n_features);

% 显示被剔除的特征
excluded_features = find(best_beta_lasso == 0);
if ~isempty(excluded_features)
    fprintf('   被剔除的特征: ');
    for i = 1:length(excluded_features)
        fprintf('%s ', feature_names{excluded_features(i)});
    end
    fprintf('\n');
end

% 保存结果
results.lasso.mse = best_lasso_mse;
results.lasso.beta = best_beta_lasso;
results.lasso.n_nonzero = best_n_nonzero_lasso;
results.lasso.beta_all = B_lasso;
results.lasso.lambda_values = lambda_values;
results.lasso.excluded_features = excluded_features;

%% 6. 结果比较
fprintf('\n6. 结果比较\n');
fprintf('=== 测试集MSE比较 ===\n');
fprintf('普通线性回归: %.4f\n', results.linear.mse);
fprintf('岭回归 (λ=%.3f): %.4f\n', best_lambda_ridge, results.ridge.mse);
fprintf('Lasso回归 (λ=%.3f): %.4f\n', best_lambda_lasso, results.lasso.mse);

fprintf('\n=== 特征稀疏性比较 ===\n');
fprintf('普通线性回归: %d/%d 非零特征\n', results.linear.n_nonzero, n_features);
fprintf('岭回归: %d/%d 非零特征\n', results.ridge.n_nonzero, n_features);
fprintf('Lasso回归: %d/%d 非零特征\n', results.lasso.n_nonzero, n_features);

%% 7. 可视化（参考示例脚本风格）
fprintf('\n7. 可视化（示例式）...\n');

% 选择两项最重要的特征用于 2D 可视化（类似示例中的 2D 约束视角）
[~, order] = sort(abs(beta_linear(2:end)), 'descend');
idx1 = order(1); idx2 = order(2);

% 构建小规模的2D问题（截距 + 两个特征）来展示损失与正则的几何
X2_train = [ones(n_train,1), X_train_std(:, [idx1, idx2])];
feature_pair_names = feature_names([idx1, idx2]);

% OLS和正则的解（用于 plotting）
beta2_ols = X2_train \ y_train_std;
I2 = eye(3); I2(1,1) = 0;
beta2_ridge = (X2_train' * X2_train + best_lambda_ridge * I2) \ (X2_train' * y_train_std);
beta2_lasso = lasso(X_train_std(:, [idx1, idx2]), y_train_std, 'Lambda', best_lambda_lasso, 'Standardize', false);
if ismatrix(beta2_lasso)
    beta2_lasso = [FitInfo.Intercept(best_lasso_idx); beta2_lasso(:,1)];
else
    beta2_lasso = [0; 0; 0];
end

% 画系数空间的损失轮廓（近似）
gridN = 60;
center1 = beta2_ols(2); center2 = beta2_ols(3);
span1 = max(1, abs(center1))*1.2; span2 = max(1, abs(center2))*1.2;
b1v = linspace(center1 - span1, center1 + span1, gridN);
b2v = linspace(center2 - span2, center2 + span2, gridN);
[B1, B2] = meshgrid(b1v, b2v);
loss_ols = zeros(size(B1)); loss_ridge = loss_ols; loss_lasso = loss_ols;

for i = 1:size(B1,1)
    for j = 1:size(B1,2)
        b = [beta2_ols(1); B1(i,j); B2(i,j)];
        y_pred = X2_train * b;
        loss_ols(i,j) = sum((y_train_std - y_pred).^2);
        loss_ridge(i,j) = loss_ols(i,j) + best_lambda_ridge * sum(b(2:end).^2);
        loss_lasso(i,j) = loss_ols(i,j) + best_lambda_lasso * sum(abs(b(2:end)));
    end
end

figure('Position', [100, 100, 1400, 450]);
subplot(1,3,1);
contour(B1, B2, loss_ols, 30); hold on;
plot(beta2_ols(2), beta2_ols(3), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
title('OLS损失（近似）');
xlabel(sprintf('%s系数', feature_pair_names{1})); ylabel(sprintf('%s系数', feature_pair_names{2}));
grid on;

subplot(1,3,2);
contour(B1, B2, loss_ridge, 30); hold on;
plot(beta2_ridge(2), beta2_ridge(3), 'rs', 'MarkerSize', 8, 'LineWidth', 2);
title(sprintf('岭回归损失 (λ=%.3f)', best_lambda_ridge));
xlabel(sprintf('%s系数', feature_pair_names{1})); ylabel(sprintf('%s系数', feature_pair_names{2}));
grid on;

subplot(1,3,3);
contour(B1, B2, loss_lasso, 30); hold on;
if exist('beta2_lasso','var') && ~isempty(beta2_lasso)
    plot(beta2_lasso(2), beta2_lasso(3), 'g^', 'MarkerSize', 8, 'LineWidth', 2);
end
title(sprintf('Lasso损失近似 (λ=%.3f)', best_lambda_lasso));
xlabel(sprintf('%s系数', feature_pair_names{1})); ylabel(sprintf('%s系数', feature_pair_names{2}));
grid on;
sgtitle('系数空间中的损失与正则化（使用两最重要特征）');
saveas(gcf, 'loss_contours_two_features.png');
fprintf('   已保存: loss_contours_two_features.png\n');

figure('Position', [100,100,1200,500]);
% 岭回归系数路径（不含截距）
subplot(1,2,1);
colors = lines(n_features);
for i = 1:n_features
    semilogx(lambda_values, results.ridge.beta_all(i+1,:), 'LineWidth', 2, 'Color', colors(i,:), 'DisplayName', feature_names{i}); hold on;
end
xlabel('Lambda (对数尺度)'); ylabel('系数值');
title('岭回归系数路径'); legend('show', 'Location', 'best', 'NumColumns', 2); grid on;

% Lasso系数路径
subplot(1,2,2);
for i = 1:n_features
    semilogx(results.lasso.lambda_values, results.lasso.beta_all(i,:), 'LineWidth', 2, 'Color', colors(i,:), 'DisplayName', feature_names{i}); hold on;
end
xlabel('Lambda (对数尺度)'); ylabel('系数值');
title('Lasso回归系数路径'); legend('show', 'Location', 'best', 'NumColumns', 2); grid on;
saveas(gcf, 'coefficient_paths.png');
fprintf('   已保存: coefficient_paths.png\n');

% --- MSE vs Lambda & 稀疏性 vs Lambda（Ridge & Lasso）
figure('Position', [100, 100, 1000, 600]);
subplot(2,1,1);
semilogx(lambda_values, mse_ridge_values, 'r-o', 'LineWidth', 2, 'DisplayName','Ridge MSE'); hold on;
semilogx(lambda_values, mse_lasso_values, 'b-s', 'LineWidth', 2, 'DisplayName','Lasso MSE');
xlabel('Lambda (对数尺度)'); ylabel('MSE'); title('MSE vs Lambda'); legend('show'); grid on;

subplot(2,1,2);
semilogx(lambda_values, n_nonzero_ridge_values, 'r--o', 'LineWidth', 2, 'DisplayName','Ridge 非零特征'); hold on;
semilogx(lambda_values, n_nonzero_lasso_values, 'b--s', 'LineWidth', 2, 'DisplayName','Lasso 非零特征');
xlabel('Lambda (对数尺度)'); ylabel('非零特征数量'); title('非零特征数 vs Lambda'); legend('show'); grid on;
saveas(gcf, 'lambda_analysis.png');
fprintf('   已保存: lambda_analysis.png\n');

% --- 预测 vs 真实（线性、岭、Lasso）
figure('Position', [100,100,1200,400]);
subplot(1,3,1);
scatter(y_test, results.linear.y_pred, 'filled'); hold on;
plot(sort(y_test), sort(y_test), 'k--'); xlabel('真实值'); ylabel('预测值'); title('线性回归：预测 vs 真实'); grid on;

subplot(1,3,2);
y_pred_ridge_best = X_test_with_intercept * results.ridge.beta * sigma_y_train + mu_y_train;
scatter(y_test, y_pred_ridge_best, 'filled'); hold on; plot(sort(y_test), sort(y_test), 'k--');
xlabel('真实值'); ylabel('预测值'); title(sprintf('岭回归 (λ=%.3f)', best_lambda_ridge)); grid on;

subplot(1,3,3);
y_pred_lasso_best = X_test_std * results.lasso.beta + results.lasso.beta'*0; % compute using saved beta and intercept

y_pred_lasso = X_test_std * results.lasso.beta + FitInfo.Intercept(best_lasso_idx);
y_pred_lasso = y_pred_lasso * sigma_y_train + mu_y_train;
scatter(y_test, y_pred_lasso, 'filled'); hold on; plot(sort(y_test), sort(y_test), 'k--');
xlabel('真实值'); ylabel('预测值'); title(sprintf('Lasso (λ=%.3f)', best_lambda_lasso)); grid on;
saveas(gcf, 'pred_vs_true.png');
fprintf('   已保存: pred_vs_true.png\n');
