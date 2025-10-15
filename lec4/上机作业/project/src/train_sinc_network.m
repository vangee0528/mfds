function results = train_sinc_network(config)
validate_config(config);
if isfield(config, 'rng_seed') && ~isempty(config.rng_seed); rng(config.rng_seed); end

N_total = config.N_total;
[inputs, targets] = generate_samples(N_total, config.domain_min, config.domain_max, config.min_radius);

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

layer_sizes = [size(inputs, 1), config.hidden_sizes, 1];
model = initialize_model(layer_sizes);

max_epochs = config.max_epochs;
learning_rate = config.learning_rate;
batch_size = config.batch_size;
num_batches = ceil(N_train / batch_size);

history.trainLoss = zeros(max_epochs, 1);
history.valLoss = zeros(max_epochs, 1);

for epoch = 1:max_epochs
    order = randperm(N_train);
    train_inputs = train_inputs(:, order);
    train_targets = train_targets(:, order);
    for b = 1:num_batches
        s = (b-1)*batch_size + 1;
        e = min(b*batch_size, N_train);
        batch_inputs = train_inputs(:, s:e);
        batch_targets = train_targets(:, s:e);
        [gradW, gradB] = compute_gradients(model, batch_inputs, batch_targets);
        model = apply_gradients(model, gradW, gradB, learning_rate);
    end
    history.trainLoss(epoch) = compute_mse(model, train_inputs, train_targets);
    history.valLoss(epoch) = compute_mse(model, val_inputs, val_targets);
end

pred_test = forward_pass(model, test_inputs);
abs_denominator = max(abs(test_targets), 1e-8);
relative_errors = abs(pred_test - test_targets) ./ abs_denominator;
test_err = mean(relative_errors);

results = struct();
results.model = model;
results.history = history;
results.testError = test_err;
results.counts = struct('total', N_total, 'train', N_train, 'val', N_val, 'test', numel(test_idx));
results.config = config;
results.predict = @(x) forward_pass(model, x);
results.trainInputs = train_inputs;
results.trainTargets = train_targets;
results.valInputs = val_inputs;
results.valTargets = val_targets;
results.testInputs = test_inputs;
results.testTargets = test_targets;
end

function [inputs, targets] = generate_samples(N_total, domain_min, domain_max, min_radius)
inputs = zeros(2, N_total);
targets = zeros(1, N_total);
generated = 0;
while generated < N_total
    remain = N_total - generated;
    candidate = domain_min + (domain_max - domain_min) * rand(2, remain);
    r_candidate = sqrt(sum(candidate.^2, 1));
    mask = r_candidate >= min_radius;
    valid = candidate(:, mask);
    count = size(valid, 2);
    if count == 0; continue; end
    inputs(:, generated + (1:count)) = valid;
    r_valid = sqrt(sum(valid.^2, 1));
    targets(:, generated + (1:count)) = sinc(r_valid / pi);
    generated = generated + count;
end
end

function model = initialize_model(layer_sizes)
num_layers = numel(layer_sizes) - 1;
model.W = cell(num_layers, 1);
model.b = cell(num_layers, 1);
for l = 1:num_layers
    fan_in = layer_sizes(l);
    fan_out = layer_sizes(l+1);
    limit = sqrt(6 / (fan_in + fan_out));
    model.W{l} = -limit + 2*limit*rand(fan_out, fan_in);
    model.b{l} = zeros(fan_out, 1);
end
end

function [gradW, gradB] = compute_gradients(model, inputs, targets)
[activations, ~] = forward_internal(model, inputs);
outputs = activations{end};
B = size(inputs, 2);
num_layers = numel(model.W);

gradW = cell(num_layers, 1);
gradB = cell(num_layers, 1);

delta = 2 * (outputs - targets) / B;
for l = num_layers:-1:1
    if l < num_layers
        delta = (model.W{l+1}' * dZ) .* (1 - activations{l+1}.^2);
    end
    dZ = delta;
    gradW{l} = dZ * activations{l}';
    gradB{l} = sum(dZ, 2);
end
end

function model = apply_gradients(model, gradW, gradB, lr)
for l = 1:numel(model.W)
    model.W{l} = model.W{l} - lr * gradW{l};
    model.b{l} = model.b{l} - lr * gradB{l};
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
        activations{l+1} = z;
    else
        activations{l+1} = tansig_custom(z);
    end
end
end

function y = tansig_custom(x)
y = 2 ./ (1 + exp(-2*x)) - 1;
end

function validate_config(config)
fields = {"N_total","domain_min","domain_max","min_radius","hidden_sizes","max_epochs","learning_rate","batch_size"};
for k = 1:numel(fields)
    if ~isfield(config, fields{k})
        error('train_sinc_network:MissingConfig','缺少字段 %s', fields{k});
    end
end
end
