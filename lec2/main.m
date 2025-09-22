% 多维多分类LDA实验
train_len = 2000;
test_len = 1000;

% 生成训练数据
train_data = generate_stamps(train_len);
X_train = train_data(:, 1:end-1);
Y_train = train_data(:, end);

% 训练LDA模型
param = fit_lda(train_data);

% 在训练集上测试
train_pred = test_lda(X_train, param);
train_acc = sum(train_pred == Y_train) / train_len;
fprintf('训练集准确率: %.4f\n', train_acc);

% 生成测试数据
test_data = generate_stamps(test_len);
X_test = test_data(:, 1:end-1);
Y_test = test_data(:, end);

% 在测试集上测试
test_pred = test_lda(X_test, param);
test_acc = sum(test_pred == Y_test) / test_len;
fprintf('测试集准确率: %.4f\n', test_acc);