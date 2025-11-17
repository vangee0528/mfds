%% project2 主脚本：比较 OLS / Ridge / Lasso
close all; clc;

projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projectRoot, 'src')));

% 参数设置
trainRatio = 0.7;
randomSeed = 42;
lambdaGrid = [0.001, 0.01, 0.1, 1, 10, 100];
kFold = 5;

dataPath = fullfile(projectRoot, '..', 'diabetes.csv');
if ~isfile(dataPath)
    dataPath = fullfile(projectRoot, 'data', 'diabetes.csv');
end

resultsDir = fullfile(projectRoot, 'results');
figureDir = fullfile(resultsDir, 'figures');
if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

% 数据准备
data = prepare_diabetes_data(dataPath, trainRatio, randomSeed);
featureNames = data.feature_names;

% 训练三种模型
linearModel = fit_linear_regression(data.X_train, data.y_train);
ridgeResult = fit_ridge_regression(data.X_train, data.y_train, lambdaGrid, kFold);
lassoResult = fit_lasso_regression(data.X_train, data.y_train, lambdaGrid, kFold);

% 评估测试集表现
modelNames = {'OLS', 'Ridge', 'Lasso'};
mses = zeros(3, 1);
nonZeroCounts = zeros(3, 1);
bestLambdas = [NaN; ridgeResult.lambda; lassoResult.lambda];

% OLS
pred = predict_regression(linearModel, data.X_test);
mses(1) = compute_mse(data.y_test, pred);
nonZeroCounts(1) = nnz(abs(linearModel.beta) > 1e-6);

% Ridge
pred = predict_regression(ridgeResult.model, data.X_test);
mses(2) = compute_mse(data.y_test, pred);
nonZeroCounts(2) = nnz(abs(ridgeResult.model.beta) > 1e-6);

% Lasso
pred = predict_regression(lassoResult.model, data.X_test);
mses(3) = compute_mse(data.y_test, pred);
nonZeroCounts(3) = nnz(abs(lassoResult.model.beta) > 1e-6);

summaryTable = table(modelNames', mses, nonZeroCounts, bestLambdas, ...
    'VariableNames', {'Model', 'TestMSE', 'NonZeroCoeffs', 'BestLambda'});

disp(summaryTable);

% 保存结果
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

summaryPath = fullfile(resultsDir, 'performance_metrics.csv');
writetable(summaryTable, summaryPath);

metricsTxt = fullfile(resultsDir, 'performance_report.txt');
fid = fopen(metricsTxt, 'w');
fprintf(fid, 'Diabetes 回归比较\n');
fprintf(fid, '====================\n');
fprintf(fid, '训练集比例: %.2f\n', trainRatio);
fprintf(fid, 'lambda 候选: %s\n\n', mat2str(lambdaGrid));
for i = 1:height(summaryTable)
    fprintf(fid, '%s -> Test MSE = %.4f, 非零系数 = %d, 最优 lambda = %s\n', ...
        summaryTable.Model{i}, summaryTable.TestMSE(i), summaryTable.NonZeroCoeffs(i), ...
        num2str(summaryTable.BestLambda(i)));
end
fclose(fid);

disp(['结果已保存至: ', metricsTxt]);

% 绘图
coefPathFig = fullfile(figureDir, 'coefficient_paths.png');
plot_coeff_paths(ridgeResult.lambdaGrid, ridgeResult.coefPath, ...
    lassoResult.lambdaGrid, lassoResult.coefPath, featureNames, coefPathFig);

performanceFig = fullfile(figureDir, 'performance_comparison.png');
plot_model_performance(modelNames, mses, nonZeroCounts, performanceFig);

% 结束提示
fprintf('\n最佳岭回归 lambda: %.4f\n', ridgeResult.lambda);
fprintf('最佳 Lasso lambda: %.4f\n', lassoResult.lambda);
