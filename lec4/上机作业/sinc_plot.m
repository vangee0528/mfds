%% 二维 sinc 函数拟合实验（自定义神经网络实现）

clc; clear;

config = struct();
config.rng_seed = 2025;
config.N_total = 8000;
config.domain_min = -8;
config.domain_max = 8;
config.min_radius = 1e-6;
config.hidden_sizes = [20 20];
config.max_epochs = 800;   % 自定义训练轮数，可按需调整
config.learning_rate = 0.01;
config.batch_size = 256;
config.grid_step = 0.25;
config.output_dir = fullfile(pwd, 'figures');

if ~exist(config.output_dir, 'dir')
	mkdir(config.output_dir);
end

results = train_sinc_network(config);

fprintf('自定义神经网络在测试集上的平均相对误差：%.6f\n', results.testError);

saved_paths = visualize_sinc_results(results, config);

disp('所有图像与结果均已保存至 figures 目录：');
disp(saved_paths);
