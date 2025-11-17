function params = project_parameters(paths)
%PROJECT_PARAMETERS Define tunable knobs used across experiments.
%
%   INPUTS:
%       paths (struct, optional): output of project_paths. When omitted,
%       the helper computes it internally so standalone scripts can call
%       the function without extra plumbing.
%
%   OUTPUTS:
%       params (struct): consolidated configuration bundle.

    if nargin < 1 || isempty(paths)
        paths = project_paths();
    end

    params = struct();
    params.train_ratio = 0.7;
    params.random_seed = 42;
    params.lambda_grid = [1e-3, 1e-2, 1e-1, 1, 10, 100];
    params.kfold = 5;
    params.standardize = true;
    params.coeff_threshold = 1e-4;

    params.files = struct();
    params.files.cleaned_data = fullfile(paths.data_processed, 'diabetes_cleaned.mat');
    params.files.train_test_split = fullfile(paths.data_processed, 'train_test_split.mat');
    params.files.performance_metrics = fullfile(paths.results_reports, 'performance_metrics.txt');
    params.files.feature_analysis = fullfile(paths.results_reports, 'feature_analysis.txt');
    params.files.summary_report = fullfile(paths.results_reports, 'summary_report.md');

    params.figures = struct();
    params.figures.coefficient_paths = fullfile(paths.results_figures, 'coefficient_paths.png');
    params.figures.mse_comparison = fullfile(paths.results_figures, 'mse_comparison.png');
    params.figures.feature_importance = fullfile(paths.results_figures, 'feature_importance.png');
end
