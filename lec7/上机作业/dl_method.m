%% MATLAB神经网络作业 - 函数逼近与导数比较
clear; clc; close all;

fprintf('=== 作业：逼近 y = x^3 - 2x + exp(-x) 并分析导数 ===\n');

targetFun = @(x) x.^3 - 2*x + exp(-x);
targetFirstDeriv = @(x) 3*x.^2 - 2 - exp(-x);
targetSecondDeriv = @(x) 6*x + exp(-x);

scriptDir = fileparts(mfilename('fullpath'));
figDir = fullfile(scriptDir, 'figures');
if ~exist(figDir, 'dir')
	mkdir(figDir);
end

xTrain = linspace(-2, 2, 200)';
yTrain = targetFun(xTrain);

configs = struct( ...
	'name', {'1x20_tanh', '2x30_tanh', '3x20_tanh', '4x30_tanh', '2x40_sigmoid', '3x30_sigmoid'}, ...
	'hiddenSizes', {[20], [30 30], [20 20 20], [30 30 30 30], [40 40], [30 30 30]}, ...
	'activation', {'tanh', 'tanh', 'tanh', 'tanh', 'sigmoid', 'sigmoid'}, ...
	'learningRate', {0.01, 0.01, 0.01, 0.01, 0.01, 0.01}, ...
	'momentum', {0.9, 0.9, 0.9, 0.9, 0.9, 0.9}, ...
	'numEpochs', {1200, 1500, 2000, 2500, 2500, 3000} ...
);

numConfigs = numel(configs);
results(numConfigs) = struct('config', [], 'net', [], 'lossHistory', [], 'trainPred', [], ...
	'trainMSE', [], 'finalLoss', []);

numCols = min(numConfigs, 2);
numRows = ceil(numConfigs / numCols);

figFits = figure(1); clf;
tlFits = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlFits, '训练集拟合对比');

figLoss = figure(2); clf;
tlLoss = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlLoss, '训练损失曲线');

for idx = 1:numConfigs
	cfg = configs(idx);
	fprintf('\n配置 %d: %s\n', idx, cfg.name);
	fprintf('隐藏层结构: %s, 激活函数: %s\n', mat2str(cfg.hiddenSizes), cfg.activation);
	fprintf('训练轮数: %d, 学习率: %.3f, 动量: %.2f\n', cfg.numEpochs, cfg.learningRate, cfg.momentum);
    
	[trainedNet, lossHistory] = trainConfiguration(cfg, xTrain, yTrain);
    
	xDl = dlarray(xTrain', 'CB');
	yPred = extractdata(forward(trainedNet, xDl))';
	trainMSE = mean((yPred - yTrain).^2);
    
	results(idx).config = cfg;
	results(idx).net = trainedNet;
	results(idx).lossHistory = lossHistory;
	results(idx).trainPred = yPred;
	results(idx).trainMSE = trainMSE;
	results(idx).finalLoss = lossHistory(end);
    
	fprintf('最终损失: %.4e, 训练集 MSE: %.4e\n', results(idx).finalLoss, trainMSE);
    
	figure(1);
	nexttile(tlFits);
	plot(xTrain, yTrain, 'k-', 'LineWidth', 1.5); hold on;
	plot(xTrain, yPred, 'r--', 'LineWidth', 1.5);
	grid on;
	ylabel('y');
	legend('目标函数', '网络预测', 'Location', 'best');
	title(sprintf('%s | MSE=%.2e', cfg.name, trainMSE));
    
	figure(2);
	nexttile(tlLoss);
	semilogy(lossHistory, 'LineWidth', 1.2);
	grid on;
	ylabel('Loss');
	xlabel('Epoch');
	title(cfg.name);
end

[~, bestIdx] = min([results.trainMSE]);
bestResult = results(bestIdx);

fprintf('\n=== 选择表现最佳配置用于导数对比：%s ===\n', bestResult.config.name);

xEval = linspace(-3, 3, 150)';
[yNet, dyNet, d2yNet] = sweepDerivatives(bestResult.net, xEval);

yTrue = targetFun(xEval);
dyTrue = targetFirstDeriv(xEval);
d2yTrue = targetSecondDeriv(xEval);

relErr1 = abs(dyNet - dyTrue) ./ max(abs(dyTrue), 1e-8);
relErr2 = abs(d2yNet - d2yTrue) ./ max(abs(d2yTrue), 1e-8);

avgErr1 = mean(relErr1);
avgErr2 = mean(relErr2);

fprintf('平均一阶导数相对误差: %.4e\n', avgErr1);
fprintf('平均二阶导数相对误差: %.4e\n', avgErr2);

figDeriv = figure(3); clf;
tlDeriv = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlDeriv, sprintf('导数比较 | %s', bestResult.config.name));

nexttile(tlDeriv);
plot(xEval, dyTrue, 'k-', 'LineWidth', 1.5); hold on;
plot(xEval, dyNet, 'b--', 'LineWidth', 1.5);
grid on;
ylabel('dy/dx');
legend('解析解', '网络', 'Location', 'best');
title('一阶导数');

nexttile(tlDeriv);
plot(xEval, d2yTrue, 'k-', 'LineWidth', 1.5); hold on;
plot(xEval, d2yNet, 'b--', 'LineWidth', 1.5);
grid on;
ylabel('d^2y/dx^2');
legend('解析解', '网络', 'Location', 'best');
title('二阶导数');

nexttile(tlDeriv);
plot(xEval, relErr1, 'r-', 'LineWidth', 1.2); hold on;
plot(xEval, relErr2, 'm--', 'LineWidth', 1.2);
grid on;
xlabel('x');
ylabel('相对误差');
legend('一阶', '二阶', 'Location', 'best');
title('相对误差分布');

saveFigure(figFits, fullfile(figDir, 'training_fits.png'));
saveFigure(figLoss, fullfile(figDir, 'training_loss.png'));
saveFigure(figDeriv, fullfile(figDir, 'derivative_comparison.png'));

%% 辅助函数
function [trainedNet, lossHistory] = trainConfiguration(config, xTrain, yTrain)
	layers = buildSequentialLayers(config.hiddenSizes, config.activation);
	net = dlnetwork(layers);
	xDl = dlarray(xTrain', 'CB');
	yDl = dlarray(yTrain', 'CB');
    
	vel = [];
	lossHistory = zeros(config.numEpochs, 1);
    
	for epoch = 1:config.numEpochs
		[loss, gradients] = dlfeval(@computeLoss, net, xDl, yDl);
		lossHistory(epoch) = extractdata(loss);
		[net, vel] = sgdmupdate(net, gradients, vel, config.learningRate, config.momentum);
	end
    
	trainedNet = net;
end

function layers = buildSequentialLayers(hiddenSizes, activationName)
	layers = [featureInputLayer(1, 'Name', 'input', 'Normalization', 'none')];
	for i = 1:numel(hiddenSizes)
		fcName = sprintf('fc%d', i);
		layers = [layers; fullyConnectedLayer(hiddenSizes(i), 'Name', fcName)];
		actName = sprintf('%s%d', activationName, i);
		switch lower(activationName)
			case 'tanh'
				actLayer = tanhLayer('Name', actName);
			case 'sigmoid'
				actLayer = sigmoidLayer('Name', actName);
			otherwise
				error('未知激活函数: %s', activationName);
		end
		layers = [layers; actLayer];
	end
	layers = [layers; fullyConnectedLayer(1, 'Name', 'output')];
end

function [loss, gradients] = computeLoss(net, x, y)
	yPred = forward(net, x);
	loss = mse(yPred, y);
	gradients = dlgradient(loss, net.Learnables);
end

function [yNet, dyNet, d2yNet] = sweepDerivatives(net, xSamples)
	numSamples = numel(xSamples);
	yNet = zeros(numSamples, 1);
	dyNet = zeros(numSamples, 1);
	d2yNet = zeros(numSamples, 1);
	for i = 1:numSamples
		xPoint = dlarray(xSamples(i));
		[yVal, dyVal, d2Val] = dlfeval(@netValueAndDerivatives, net, xPoint);
		yNet(i) = extractdata(yVal);
		dyNet(i) = extractdata(dyVal);
		d2yNet(i) = extractdata(d2Val);
	end
end

function [output, derivative, secondDerivative] = netValueAndDerivatives(net, x)
	output = forward(net, x);
	derivative = dlgradient(output, x, 'EnableHigherDerivatives', true);
	secondDerivative = dlgradient(derivative, x);
end

function saveFigure(figHandle, filePath)
	try
		exportgraphics(figHandle, filePath, 'Resolution', 300);
	catch
		saveas(figHandle, filePath);
	end
end
