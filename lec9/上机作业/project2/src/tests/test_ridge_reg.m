function tests_passed = test_ridge_reg()
%TEST_RIDGE_REG Validate ridge solution matches manual closed-form result.
    script_dir = fileparts(mfilename('fullpath'));
    bootstrap_project_environment(script_dir);

    X = (1:4)';
    y = 2 * X + 1;
    lambda = 0.5;

    model = ridge_regression(X, y, lambda);

    X_aug = [ones(size(X, 1), 1), X];
    reg = lambda * eye(2); reg(1, 1) = 0;
    theta = (X_aug' * X_aug + reg) \ (X_aug' * y);

    assert(abs(model.intercept - theta(1)) < 1e-10, 'Ridge intercept mismatch');
    assert(abs(model.beta - theta(2)) < 1e-10, 'Ridge slope mismatch');

    y_pred = model.predict(X);
    assert(all(abs(y_pred - (model.intercept + model.beta * X)) < 1e-12), ...
        'Predict handle inconsistent with closed form');

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
