% 定义参数
train_len = 20000;
test_len = 10000;
M = 5; % 特征维度
K = 4; % 类别数

% 生成训练数据
train_data = generate_stamps(train_len, M, K);

% 训练LDA模型
param = fit_lda(train_data, K);

% 在训练集上测试
train_X = train_data(:, 1:M);
train_Y = train_data(:, M+1);
train_pred = test_lda(train_X, param, K);
train_acc = sum(train_pred == train_Y) / length(train_Y);
fprintf('训练集准确率: %.4f\n', train_acc);

% 生成测试数据
test_data = generate_stamps(test_len, M, K);

% 在测试集上测试
test_X = test_data(:, 1:M);
test_Y = test_data(:, M+1);
test_pred = test_lda(test_X, param, K);
test_acc = sum(test_pred == test_Y) / length(test_Y);
fprintf('测试集准确率: %.4f\n', test_acc);