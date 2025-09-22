train_len = 20000;

% load data
train_data = generate_stamps(train_len);
param = fit_lda(train_data);
train_pred = test_lda(train_data(:,1), param);
train_acc = sum(train_pred == train_data(:,2)) / train_len;
fprintf('训练集准确率: %.4f\n', train_acc);

test_len = 10000;
test_data = generate_stamps(test_len);
test_X = test_data(:,1);
test_Y = test_data(:,2);

test_pred = test_lda(test_X, param);
test_acc = sum(test_pred == test_Y) / test_len;
fprintf('测试集准确率: %.4f\n', test_acc);