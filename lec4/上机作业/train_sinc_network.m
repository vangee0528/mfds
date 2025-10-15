function results = train_sinc_network(config)
%TRAIN_SINC_NETWORK 自定义两层隐藏层前馈网络训练二维 sinc 函数
%
%   results = TRAIN_SINC_NETWORK(config) 根据配置结构体 config 生成数据、
%   构建并训练一个前馈神经网络，返回训练后的模型参数、训练曲线以及
%   测试集指标。整个流程未使用 MATLAB 神经网络工具箱，所有计算均基
%   于矩阵运算与自定义反向传播实现。
%
%   config 必须包含以下字段：
%       N_total      - 样本总数 (<= 10000)
%       domain_min   - 输入最小值
%       domain_max   - 输入最大值
%       min_radius   - 避免除零的最小半径
%       hidden_sizes - 隐藏层维度数组，例如 [20 20]
%       max_epochs   - 最大训练轮数
%       learning_rate- 学习率
%       batch_size   - 单次梯度更新的样本数
%
%   可选字段：
%       rng_seed     - 随机数种子
%
%   返回的 results 结构体包含：
%       model        - 网络权重与偏置
%       history      - 训练/验证损失曲线
%       testError    - 测试集平均相对误差
%       counts       - 数据集规模统计
%       predict      - 预测函数句柄
%
%   作者：自定义实现，无 MATLAB NN 工具箱依赖。

validate_config(config);

if isfield(config, 'rng_seed') && ~isempty(config.rng_seed)
    rng(config.rng_seed);
end

% === 数据生成 ===
N_total    = config.N_total;
domain_min = config.domain_min;
domain_max = config.domain_max;
min_radius = config.min_radius;

[inputs, targets] = generate_samples(N_total, domain_min, domain_max, min_radius);

% === 数据划分 ===
N_train = round(0.7 * N_total);
N_val   = round(0.15 * N_total);
idx = randperm(N_total);
train_idx = idx(1:N_train);
val_idx   = idx(N_train+1 : N_train+N_val);
test_idx  = idx(N_train+N_val+1 : end);

train_inputs = inputs(:, train_idx);
train_targets = targets(:, train_idx);
val_inputs = inputs(:, val_idx);
val_targets = targets(:, val_idx);
test_inputs = inputs(:, test_idx);
test_targets = targets(:, test_idx);

% === 模型初始化 ===
layer_sizes = [size(inputs, 1), config.hidden_sizes, 1];
model = initialize_model(layer_sizes);

% === 训练 ===
max_epochs    = config.max_epochs;
learning_rate = config.learning_rate;
batch_size    = config.batch_size;
num_batches   = ceil(N_train / batch_size);

history.trainLoss = zeros(max_epochs, 1);
history.valLoss   = zeros(max_epochs, 1);

for epoch = 1:max_epochs
    perm = randperm(N_train);
    train_inputs = train_inputs(:, perm);
    train_targets = train_targets(:, perm);

    for b = 1:num_batches
        start_idx = (b-1)*batch_size + 1;
        end_idx   = min(b*batch_size, N_train);
        batch_inputs  = train_inputs(:, start_idx:end_idx);
        batch_targets = train_targets(:, start_idx:end_idx);

        [gradW, gradB] = compute_gradients(model, batch_inputs, batch_targets);
        model = apply_gradients(model, gradW, gradB, learning_rate);
    end

    history.trainLoss(epoch) = compute_mse(model, train_inputs, train_targets);
    history.valLoss(epoch)   = compute_mse(model, val_inputs, val_targets);
end

% === 测试评估 ===
pred_test = forward_pass(model, test_inputs);
abs_denominator = max(abs(test_targets), 1e-8);
relative_errors = abs(pred_test - test_targets) ./ abs_denominator;
test_err = mean(relative_errors);

% === 结果封装 ===
results = struct();
results.model = model;
results.history = history;
results.testError = test_err;
results.counts = struct('total', N_total, 'train', N_train, ...
                        'val', N_val, 'test', numel(test_idx));
results.config = config;

    function preds = predictor(inputs_to_predict)
        preds = forward_pass(model, inputs_to_predict);
    end

results.predict = @predictor;

% 附带训练数据以便可视化或复现
results.trainInputs = train_inputs;
results.trainTargets = train_targets;
results.valInputs = val_inputs;
results.valTargets = val_targets;
results.testInputs = test_inputs;
results.testTargets = test_targets;

end

% ==================== 辅助函数 ====================

function [inputs, targets] = generate_samples(N_total, domain_min, domain_max, min_radius)
num_features = 2;
inputs  = zeros(num_features, N_total);
targets = zeros(1, N_total);

generated = 0;
while generated < N_total
    remain    = N_total - generated;
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
    targets(:, generated + (1:num_valid)) = sinc(r_valid / pi);
    generated = generated + num_valid;
end
end

function model = initialize_model(layer_sizes)
num_layers = numel(layer_sizes) - 1;
model.W = cell(num_layers, 1);
model.b = cell(num_layers, 1);

for l = 1:num_layers
    fan_in  = layer_sizes(l);
    fan_out = layer_sizes(l+1);
    % Xavier 初始化
    limit = sqrt(6 / (fan_in + fan_out));
    model.W{l} = -limit + 2*limit*rand(fan_out, fan_in);
    model.b{l} = zeros(fan_out, 1);
end
end

function [gradW, gradB] = compute_gradients(model, inputs, targets)
[activations, preactivations] = forward_internal(model, inputs);
outputs = activations{end};
B = size(inputs, 2);

delta = 2 * (outputs - targets) / B; % dL/dOutputs
num_layers = numel(model.W);

gradW = cell(num_layers, 1);
gradB = cell(num_layers, 1);

for l = num_layers:-1:1
    if l == num_layers
        dZ = delta; % 输出层线性
    else
        dZ = delta .* (1 - activations{l+1}.^2); % tanh 导数
    end

    gradW{l} = dZ * activations{l}';
    gradB{l} = sum(dZ, 2);

    if l > 1
        delta = model.W{l}' * dZ;
    end
end
end

function model = apply_gradients(model, gradW, gradB, learning_rate)
for l = 1:numel(model.W)
    model.W{l} = model.W{l} - learning_rate * gradW{l};
    model.b{l} = model.b{l} - learning_rate * gradB{l};
end
end

function loss = compute_mse(model, inputs, targets)
outputs = forward_pass(model, inputs);
loss = mean((outputs - targets).^2);
end

function outputs = forward_pass(model, inputs)
[activations, ~] = forward_internal(model, inputs);
outputs = activations{end};
end

function [activations, preactivations] = forward_internal(model, inputs)
num_layers = numel(model.W);
activations = cell(num_layers + 1, 1);
preactivations = cell(num_layers, 1);

activations{1} = inputs;

for l = 1:num_layers
    z = model.W{l} * activations{l} + model.b{l};
    preactivations{l} = z;
    if l == num_layers
        activations{l+1} = z; % 输出层线性
    else
        activations{l+1} = tansig_custom(z);
    end
end
end

function y = tansig_custom(x)
y = 2 ./ (1 + exp(-2*x)) - 1;
end

function validate_config(config)
required = {"N_total", "domain_min", "domain_max", "min_radius", ...
            "hidden_sizes", "max_epochs", "learning_rate", "batch_size"};
for k = 1:numel(required)
    if ~isfield(config, required{k})
        error('train_sinc_network:MissingConfig', ...
              '配置结构缺少字段 "%s"。', required{k});
    end
end
end
