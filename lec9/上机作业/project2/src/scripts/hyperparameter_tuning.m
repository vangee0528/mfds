function tuning = hyperparameter_tuning(split, params)
%HYPERPARAMETER_TUNING Cross-validation search for ridge and Lasso models.
%
%   INPUTS:
%       split  - struct returned by DATA_PREPROCESSOR with train data.
%       params - struct returned by PROJECT_PARAMETERS containing lambda_grid
%                and kfold fields.
%
%   OUTPUTS:
%       tuning struct with sub-structures for ridge and lasso containing:
%           best_lambda        - lambda with smallest CV error
%           cv_mse             - vector of averaged fold MSEs per lambda
%           coefficient_path   - coefficients (features x lambdas)
%           model              - trained model on full training data
%           lambda_grid        - lambda candidates (ascending)
%
%   The helper performs k-fold cross validation over the provided lambda
%   grid and re-trains models on all training samples using the best lambda.

    arguments
        split struct
        params struct
    end

    lambda_grid = params.lambda_grid(:)';
    num_lambdas = numel(lambda_grid);
    n_features = size(split.X_train, 2);

    cv = cvpartition(size(split.X_train, 1), 'KFold', params.kfold);

    ridge_cv_mse = zeros(1, num_lambdas);
    lasso_cv_mse = zeros(1, num_lambdas);

    for li = 1:num_lambdas
        lambda = lambda_grid(li);
        ridge_fold_mse = zeros(cv.NumTestSets, 1);
        lasso_fold_mse = zeros(cv.NumTestSets, 1);

        for fold = 1:cv.NumTestSets
            train_idx = training(cv, fold);
            val_idx = test(cv, fold);

            X_train_fold = split.X_train(train_idx, :);
            y_train_fold = split.y_train(train_idx);
            X_val_fold = split.X_train(val_idx, :);
            y_val_fold = split.y_train(val_idx);

            ridge_model = ridge_regression(X_train_fold, y_train_fold, lambda);
            ridge_preds = ridge_model.predict(X_val_fold);
            ridge_fold_mse(fold) = mean((y_val_fold - ridge_preds).^2);

            lasso_model = lasso_regression(X_train_fold, y_train_fold, lambda);
            lasso_preds = lasso_model.predict(X_val_fold);
            lasso_fold_mse(fold) = mean((y_val_fold - lasso_preds).^2);
        end

        ridge_cv_mse(li) = mean(ridge_fold_mse);
        lasso_cv_mse(li) = mean(lasso_fold_mse);
    end

    [~, ridge_best_idx] = min(ridge_cv_mse);
    [~, lasso_best_idx] = min(lasso_cv_mse);

    ridge_best_lambda = lambda_grid(ridge_best_idx);
    lasso_best_lambda = lambda_grid(lasso_best_idx);

    ridge_best_model = ridge_regression(split.X_train, split.y_train, ridge_best_lambda);
    lasso_best_model = lasso_regression(split.X_train, split.y_train, lasso_best_lambda);

    ridge_coeff_path = zeros(n_features, num_lambdas);
    for li = 1:num_lambdas
        model = ridge_regression(split.X_train, split.y_train, lambda_grid(li));
        ridge_coeff_path(:, li) = model.beta;
    end

    % MATLAB's lasso expects lambda values in descending order, so keep a
    % copy for plotting while returning in ascending order to the caller.
    [lambda_sorted_desc, sort_idx] = sort(lambda_grid, 'descend');
    [lasso_path_desc, fit_info] = lasso(split.X_train, split.y_train, ...
        'Lambda', lambda_sorted_desc, 'Standardize', false);
    [~, revert_idx] = sort(sort_idx, 'ascend');
    lasso_coeff_path = lasso_path_desc(:, revert_idx);

    tuning = struct();
    tuning.ridge = struct();
    tuning.ridge.best_lambda = ridge_best_lambda;
    tuning.ridge.cv_mse = ridge_cv_mse;
    tuning.ridge.coefficient_path = ridge_coeff_path;
    tuning.ridge.model = ridge_best_model;
    tuning.ridge.lambda_grid = lambda_grid;

    tuning.lasso = struct();
    tuning.lasso.best_lambda = lasso_best_lambda;
    tuning.lasso.cv_mse = lasso_cv_mse;
    tuning.lasso.coefficient_path = lasso_coeff_path;
    tuning.lasso.model = lasso_best_model;
    tuning.lasso.lambda_grid = lambda_grid;
    tuning.lasso.fit_info = fit_info;
end
