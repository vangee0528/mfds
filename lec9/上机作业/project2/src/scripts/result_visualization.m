function result_visualization()
%RESULT_VISUALIZATION Regenerate plots from saved metrics & tuning summary.
%
%   Run after MAIN_ANALYSIS to refresh figures without re-training models.

    paths = project_paths();
    params = project_parameters(paths);
    ensure_directories(paths);

    if ~isfile(params.files.performance_metrics)
        error('result_visualization:MissingMetrics', ...
            'Expected performance metrics at %s. Run main_analysis first.', ...
            params.files.performance_metrics);
    end

    tuning_summary_file = fullfile(paths.results_reports, 'tuning_summary.mat');
    if ~isfile(tuning_summary_file)
        error('result_visualization:MissingTuning', ...
            'Could not find %s. Run main_analysis to generate tuning summary.', ...
            tuning_summary_file);
    end

    metrics_table = readtable(params.files.performance_metrics, ...
        'FileType', 'text', 'Delimiter', '\t');
    load(tuning_summary_file, 'tuning'); %#ok<LOAD>

    feature_names = derive_feature_names(params, size(tuning.ridge.coefficient_path, 1));

    plots = plot_utilities();
    plots.mse_comparison(metrics_table.Model, metrics_table.MSE, params.figures.mse_comparison);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 500]);
    subplot(1, 2, 1);
    semilogx(tuning.ridge.lambda_grid, tuning.ridge.coefficient_path', 'LineWidth', 1.2);
    xlabel('\lambda'); ylabel('Coefficient'); title('Ridge Coefficient Paths'); grid on;

    subplot(1, 2, 2);
    semilogx(tuning.lasso.lambda_grid, tuning.lasso.coefficient_path', 'LineWidth', 1.2);
    xlabel('\lambda'); ylabel('Coefficient'); title('Lasso Coefficient Paths'); grid on;
    legend(feature_names, 'Interpreter', 'none', 'Location', 'bestoutside');
    exportgraphics(fig, params.figures.coefficient_paths, 'Resolution', 300);

    plots.feature_importance(feature_names, tuning.lasso.model.beta, params.figures.feature_importance);

    fprintf('Figures regenerated under %s.\n', paths.results_figures);
end

function ensure_directories(paths)
    dirs = {paths.results, paths.results_figures, paths.results_performance_figures, ...
            paths.results_models, paths.results_reports};
    for i = 1:numel(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
        end
    end
end

function feature_names = derive_feature_names(params, n_features)
    if isfield(params.files, 'train_test_split') && isfile(params.files.train_test_split)
        data = load(params.files.train_test_split, 'train_test_split');
        if isfield(data, 'train_test_split') && isfield(data.train_test_split, 'feature_names')
            feature_names = data.train_test_split.feature_names;
            return;
        end
    end
    feature_names = arrayfun(@(i) sprintf('Feature %d', i), 1:n_features, 'UniformOutput', false);
end
