function tests_passed = test_linear_reg()
%TEST_LINEAR_REG Basic sanity checks for the OLS implementation.
    script_dir = fileparts(mfilename('fullpath'));
    bootstrap_project_environment(script_dir);

    X = (1:5)';
    y = 3 * X + 2;

    model = linear_regression(X, y);
    y_pred = model.predict(X);

    assert(abs(model.intercept - 2) < 1e-10, 'Incorrect intercept for perfect linear data');
    assert(norm(model.beta - 3) < 1e-10, 'Incorrect slope for perfect linear data');
    assert(max(abs(y - y_pred)) < 1e-10, 'Predictions should perfectly fit training data');

    tests_passed = true;
end

function project_root = bootstrap_project_environment(script_dir)
    project_root = fileparts(fileparts(script_dir));
    addpath_if_missing(project_root);
    addpath_if_missing(fullfile(project_root, 'src'));
    addpath_if_missing(fullfile(project_root, 'config'));
end

function addpath_if_missing(target_path)
    if ~isfolder(target_path)
        warning('Path %s does not exist and cannot be added.', target_path);
        return;
    end
    path_entries = strsplit(path, pathsep);
    if ~any(strcmp(path_entries, target_path))
        addpath(target_path);
    end
end
