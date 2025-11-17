function paths = project_paths()
%PROJECT_PATHS Assemble and expose canonical directories for project2.
%
%   RETURNS:
%       paths (struct): contains absolute paths for key project folders.
%
%   The helper centralizes all filesystem references so that scripts can
%   remain agnostic to where the project is checked out.

    config_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(config_dir);

    paths = struct();
    paths.root = project_root;

    paths.data = fullfile(project_root, 'data');
    paths.data_raw = fullfile(paths.data, 'raw');
    paths.data_processed = fullfile(paths.data, 'processed');

    paths.src = fullfile(project_root, 'src');
    paths.utils = fullfile(paths.src, 'utils');
    paths.models = fullfile(paths.src, 'models');
    paths.scripts = fullfile(paths.src, 'scripts');
    paths.tests = fullfile(paths.src, 'tests');

    paths.results = fullfile(project_root, 'results');
    paths.results_figures = fullfile(paths.results, 'figures');
    paths.results_performance_figures = fullfile(paths.results_figures, 'performance_plots');
    paths.results_models = fullfile(paths.results, 'models');
    paths.results_reports = fullfile(paths.results, 'reports');

    paths.docs = fullfile(project_root, 'docs');
    paths.config = config_dir;

    paths.project_readme = fullfile(project_root, 'PROJECT_README.md');
end
