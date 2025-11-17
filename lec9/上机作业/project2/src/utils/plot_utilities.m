function plot_handles = plot_utilities()
%PLOT_UTILITIES Return a bundle of plotting helper closures.
%
%   Each handle provides a consistently styled plot for reuse across
%   scripts. Using nested functions keeps state encapsulated.

    plot_handles = struct();
    plot_handles.coefficient_paths = @coefficient_paths_plot;
    plot_handles.mse_comparison = @mse_comparison_plot;
    plot_handles.feature_importance = @feature_importance_plot;
end

function fig = coefficient_paths_plot(lambda_grid, coefficients, feature_names, output_path)
%COEFFICIENT_PATHS_PLOT Visualize coefficient trajectories vs lambda.
    fig = figure('Visible', 'off');
    semilogx(lambda_grid, coefficients', 'LineWidth', 1.2);
    xlabel('\lambda');
    ylabel('Coefficient value');
    title('Coefficient Path Across Regularization Strengths');
    grid on;
    legend(feature_names, 'Interpreter', 'none', 'Location', 'bestoutside');
    if nargin > 3 && ~isempty(output_path)
        exportgraphics(fig, output_path, 'Resolution', 300);
    end
end

function fig = mse_comparison_plot(model_names, mse_values, output_path)
%MSE_COMPARISON_PLOT Draw a bar chart comparing model MSEs.
    fig = figure('Visible', 'off');
    bar(categorical(model_names), mse_values);
    ylabel('Mean Squared Error');
    title('Model Performance on Test Set');
    grid on;
    if nargin > 2 && ~isempty(output_path)
        exportgraphics(fig, output_path, 'Resolution', 300);
    end
end

function fig = feature_importance_plot(feature_names, coefficients, output_path)
%FEATURE_IMPORTANCE_PLOT Show absolute coefficient magnitudes per feature.
    fig = figure('Visible', 'off');
    stem(categorical(feature_names), abs(coefficients), 'filled');
    ylabel('|Coefficient|');
    title('Feature Importance (Magnitude of Coefficients)');
    grid on;
    if nargin > 2 && ~isempty(output_path)
        exportgraphics(fig, output_path, 'Resolution', 300);
    end
end
