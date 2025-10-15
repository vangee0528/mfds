%% 二维 sinc 函数拟合主程序
clc; clear;

project_root = fileparts(fileparts(mfilename('fullpath')));
output_dir = fullfile(project_root, 'output', 'figures');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

config = struct();
config.rng_seed = 2025;
config.N_total = 8000;
config.domain_min = -8;
config.domain_max = 8;
config.min_radius = 1e-6;
config.hidden_sizes = [20 20];
config.max_epochs = 1000;
config.learning_rate = 0.01;
config.batch_size = 256;
config.grid_step = 0.25;
config.output_dir = output_dir;

results = train_sinc_network(config);

fprintf('测试集平均相对误差: %.6f\n', results.testError);

saved_paths = visualize_sinc_results(results, config);
disp(saved_paths);
