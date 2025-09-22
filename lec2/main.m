% 多维多分类LDA实验主程序
clear; clc;

% 实验参数设置
train_len = 2000;   % 训练样本数
test_len = 1000;    % 测试样本数
M = 3;              % 特征维度
K = 4;              % 类别数

fprintf('=== 多维多分类LDA实验 ===\n');
fprintf('特征维度: %d\n', M);
fprintf('类别数: %d\n', K);
fprintf('训练样本数: %d\n', train_len);
fprintf('测试样本数: %d\n\n', test_len);

% 生成训练数据
fprintf('正在生成训练数据...\n');
train_data = generate_stamps(train_len, M, K);

% 提取特征和标签
train_X = train_data(:, 1:M);
train_Y = train_data(:, M+1);

% 显示数据分布信息
fprintf('训练数据分布:\n');
unique_labels = unique(train_Y);
for i = 1:length(unique_labels)
    label = unique_labels(i);
    count = sum(train_Y == label);
    fprintf('  类别 %d: %d 个样本 (%.1f%%)\n', label, count, count/train_len*100);
end
fprintf('\n');

% 训练LDA模型
fprintf('正在训练LDA模型...\n');
param = fit_lda(train_data);
fprintf('\n');

% 在训练集上测试
fprintf('在训练集上测试...\n');
train_pred = test_lda(train_X, param);
train_acc = sum(train_pred == train_Y) / train_len;
fprintf('训练集准确率: %.4f (%.2f%%)\n\n', train_acc, train_acc*100);

% 生成测试数据
fprintf('正在生成测试数据...\n');
test_data = generate_stamps(test_len, M, K);
test_X = test_data(:, 1:M);
test_Y = test_data(:, M+1);

% 在测试集上测试
fprintf('在测试集上测试...\n');
test_pred = test_lda(test_X, param);
test_acc = sum(test_pred == test_Y) / test_len;
fprintf('测试集准确率: %.4f (%.2f%%)\n\n', test_acc, test_acc*100);

% 计算混淆矩阵
fprintf('=== 测试集混淆矩阵 ===\n');
confusion_matrix = zeros(K, K);
for i = 1:length(test_Y)
    true_label = test_Y(i) + 1;    % 转换为1-based索引
    pred_label = test_pred(i) + 1;
    confusion_matrix(true_label, pred_label) = confusion_matrix(true_label, pred_label) + 1;
end

fprintf('       预测\n');
fprintf('真实   ');
for j = 1:K
    fprintf('  %d  ', j-1);
end
fprintf('\n');
for i = 1:K
    fprintf('  %d    ', i-1);
    for j = 1:K
        fprintf('%4d ', confusion_matrix(i, j));
    end
    fprintf('\n');
end
fprintf('\n');

% 计算每个类别的精确率和召回率
fprintf('=== 各类别性能指标 ===\n');
for k = 1:K
    label = k - 1;
    TP = confusion_matrix(k, k);
    FP = sum(confusion_matrix(:, k)) - TP;
    FN = sum(confusion_matrix(k, :)) - TP;
    
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * precision * recall / (precision + recall);
    
    fprintf('类别 %d: 精确率=%.4f, 召回率=%.4f, F1分数=%.4f\n', ...
            label, precision, recall, f1_score);
end

% 可视化结果（仅当特征维度为2时）
if M == 2
    fprintf('\n正在生成可视化图像...\n');
    
    figure('Name', '多维多分类LDA结果可视化', 'NumberTitle', 'off');
    
    % 子图1: 训练数据
    subplot(2, 2, 1);
    colors = lines(K);
    for k = 1:K
        label = unique_labels(k);
        mask = (train_Y == label);
        scatter(train_X(mask, 1), train_X(mask, 2), 50, colors(k, :), 'filled');
        hold on;
    end
    title('训练数据分布');
    xlabel('特征1');
    ylabel('特征2');
    legend(arrayfun(@(x) sprintf('类别 %d', x), unique_labels, 'UniformOutput', false));
    grid on;
    
    % 子图2: 测试数据（真实标签）
    subplot(2, 2, 2);
    for k = 1:K
        label = unique_labels(k);
        mask = (test_Y == label);
        scatter(test_X(mask, 1), test_X(mask, 2), 50, colors(k, :), 'filled');
        hold on;
    end
    title('测试数据分布（真实标签）');
    xlabel('特征1');
    ylabel('特征2');
    legend(arrayfun(@(x) sprintf('类别 %d', x), unique_labels, 'UniformOutput', false));
    grid on;
    
    % 子图3: 测试数据（预测标签）
    subplot(2, 2, 3);
    for k = 1:K
        label = unique_labels(k);
        mask = (test_pred == label);
        scatter(test_X(mask, 1), test_X(mask, 2), 50, colors(k, :), 'filled');
        hold on;
    end
    title(sprintf('测试数据分布（预测标签），准确率=%.2f%%', test_acc*100));
    xlabel('特征1');
    ylabel('特征2');
    legend(arrayfun(@(x) sprintf('类别 %d', x), unique_labels, 'UniformOutput', false));
    grid on;
    
    % 子图4: 错误分类的点
    subplot(2, 2, 4);
    correct_mask = (test_pred == test_Y);
    scatter(test_X(correct_mask, 1), test_X(correct_mask, 2), 30, [0.7 0.7 0.7], 'filled');
    hold on;
    scatter(test_X(~correct_mask, 1), test_X(~correct_mask, 2), 50, 'r', 'x', 'LineWidth', 2);
    title(sprintf('分类错误的样本 (%d个)', sum(~correct_mask)));
    xlabel('特征1');
    ylabel('特征2');
    legend({'正确分类', '错误分类'});
    grid on;
    
    % 调整子图布局
    sgtitle('多维多分类LDA实验结果');
end

fprintf('\n实验完成！\n');