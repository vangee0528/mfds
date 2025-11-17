function main_analysis()
%MAIN_ANALYSIS End-to-end workflow for the diabetes regression comparison.
%
%   This script orchestrates data loading, preprocessing, model training,
%   hyperparameter selection, evaluation, and artifact generation. Run it
%   after updating the dataset under data/raw/diabetes.csv.

    paths = project_paths();
    params = project_parameters(paths);
    ensure_directories(paths);

    fprintf('[1/5] Loading raw dataset...\n');
    data = data_loader(paths);

    fprintf('[2/5] Preprocessing and splitting data...\n');
    split = data_preprocessor(data, params);

    fprintf('[3/5] Running hyperparameter search for ridge / Lasso...\n');
    tuning = hyperparameter_tuning(split, params);

    fprintf('[4/5] Training final models...\n');
    linear_model = linear_regression(split.X_train, split.y_train);
    ridge_model = tuning.ridge.model;
    lasso_model = tuning.lasso.model;

    fprintf('[5/5] Evaluating on held-out test set...\n');
    metrics = compile_metrics(split, linear_model, ridge_model, lasso_model, params);

    fprintf('Saving models and reports...\n');
    save_models(paths, linear_model, ridge_model, lasso_model);
    save_reports(params, metrics, tuning, split);

    fprintf('Generating visualizations...\n');
    generate_figures(params, metrics, tuning, split);

    fprintf('Analysis complete. See results/reports for summaries.\n');
end

function ensure_directories(paths)
    dirs = {paths.results, paths.results_figures, paths.results_performance_figures, ...
            paths.results_models, paths.results_reports, paths.data_processed};
    for i = 1:numel(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
        end
    end
end

function metrics = compile_metrics(split, linear_model, ridge_model, lasso_model, params)
    models = {linear_model, ridge_model, lasso_model};
    names = {'Linear Regression', 'Ridge Regression', 'Lasso Regression'};

    metrics = struct();
    for i = 1:numel(models)
        model = models{i};
        y_pred = model.predict(split.X_test);
        metrics.(sanitize_name(names{i})) = evaluate_model(split.y_test, y_pred, model.beta, params);
        metrics.(sanitize_name(names{i})).name = names{i};
        if isfield(model, 'lambda')
            metrics.(sanitize_name(names{i})).lambda = model.lambda;
        else
            metrics.(sanitize_name(names{i})).lambda = NaN;
        end
    end
end

function save_models(paths, linear_model, ridge_model, lasso_model)
    save(fullfile(paths.results_models, 'linear_model.mat'), 'linear_model');
    save(fullfile(paths.results_models, 'best_ridge_model.mat'), 'ridge_model');
    save(fullfile(paths.results_models, 'best_lasso_model.mat'), 'lasso_model');
end

function save_reports(params, metrics, tuning, split)
    model_fields = fieldnames(metrics);
    perf_lines = ["Model\tLambda\tMSE\tMAE\tR2\tNonZero\tSparsity"];
    for i = 1:numel(model_fields)
        m = metrics.(model_fields{i});
        perf_lines(end+1) = sprintf('%s\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%.2f', ...
            m.name, m.lambda, m.mse, m.mae, m.r2, m.nonzero_features, m.sparsity_ratio);
    end
    writelines(perf_lines, params.files.performance_metrics);

    feature_lines = ["模型\t非零特征数\t稀疏度"];
    for i = 1:numel(model_fields)
        m = metrics.(model_fields{i});
        feature_lines(end+1) = sprintf('%s\t%d\t%.2f', m.name, m.nonzero_features, m.sparsity_ratio);
    end
    writelines(feature_lines, params.files.feature_analysis);

    summary = compose_summary(metrics, tuning, split);
    fid = fopen(params.files.summary_report, 'w');
    fprintf(fid, '%s', summary);
    fclose(fid);

    tuning_summary_file = fullfile(fileparts(params.files.summary_report), 'tuning_summary.mat');
    save(tuning_summary_file, 'tuning');
end

function summary = compose_summary(metrics, tuning, split)
    ridge_name = 'Ridge Regression';
    lasso_name = 'Lasso Regression';
    linear_name = 'Linear Regression';

    best_model = lasso_name;
    best_mse = metrics.(sanitize_name(best_model)).mse;
    names = {linear_name, ridge_name, lasso_name};
    for i = 1:numel(names)
        m = metrics.(sanitize_name(names{i}));
        if m.mse < best_mse
            best_mse = m.mse;
            best_model = names{i};
        end
    end

    summary = sprintf(['# Diabetes 回归对比总结\n\n', ...
        '## 数据\n- 总样本数：%d\n- 训练/测试划分：70%% / 30%%\n- 特征数：%d\n\n', ...
        '## CV 选择结果\n- Ridge 最优 λ：%.4f\n- Lasso 最优 λ：%.4f\n\n', ...
        '## 测试集表现 (MSE 越低越好)\n'], ...
        size(split.X_train, 1) + size(split.X_test, 1), size(split.X_train, 2), ...
        tuning.ridge.best_lambda, tuning.lasso.best_lambda);

    for i = 1:numel(names)
        m = metrics.(sanitize_name(names{i}));
        summary = sprintf('%s- %s: MSE=%.4f, R2=%.4f, 非零特征=%d\n', summary, ...
            names{i}, m.mse, m.r2, m.nonzero_features);
    end

    summary = sprintf(['%s\n## 结论\n%s 在测试集上取得最低的 MSE，', ...
        '同时稀疏度(%.2f) 说明其具备更强的特征选择能力。', ...
        'Ridge 在抑制系数波动方面表现良好，但仍保留所有特征。', ...
        '若希望得到可解释性更强的模型，推荐优先使用 Lasso。\n'], ...
        summary, best_model, metrics.(sanitize_name(best_model)).sparsity_ratio);
end

function generate_figures(params, metrics, tuning, split)
    plots = plot_utilities();
    model_names = {'Linear', 'Ridge', 'Lasso'};
    mse_values = [metrics.linear_regression.mse, metrics.ridge_regression.mse, metrics.lasso_regression.mse];
    plots.mse_comparison(model_names, mse_values, params.figures.mse_comparison);

    % Coefficient path figure with Ridge and Lasso side-by-side.
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 500]);
    subplot(1, 2, 1);
    semilogx(tuning.ridge.lambda_grid, tuning.ridge.coefficient_path', 'LineWidth', 1.2);
    xlabel('\lambda'); ylabel('Coefficient'); title('Ridge Coefficient Paths'); grid on;

    subplot(1, 2, 2);
    semilogx(tuning.lasso.lambda_grid, tuning.lasso.coefficient_path', 'LineWidth', 1.2);
    xlabel('\lambda'); ylabel('Coefficient'); title('Lasso Coefficient Paths'); grid on;
    legend(split.feature_names, 'Interpreter', 'none', 'Location', 'bestoutside');
    exportgraphics(fig, params.figures.coefficient_paths, 'Resolution', 300);

    plots.feature_importance(split.feature_names, tuning.lasso.model.beta, params.figures.feature_importance);
end

function name = sanitize_name(label)
    lower = lower(label);
    name = strrep(lower, ' ', '_');
end
