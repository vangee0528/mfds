function data = data_loader(paths)
%DATA_LOADER Read the diabetes.csv dataset into MATLAB-friendly structs.
%
%   INPUTS:
%       paths (struct): output from project_paths containing at least the
%       data_raw directory.
%
%   OUTPUTS:
%       data (struct) with fields:
%           X  - numeric matrix of predictors (observations x features)
%           y  - numeric vector of targets
%           feature_names - cell array of feature labels
%           target_name   - string with the target column name

    arguments
        paths struct
    end

    dataset_path = fullfile(paths.data_raw, 'diabetes.csv');
    if ~isfile(dataset_path)
        error('data_loader:MissingFile', 'Dataset not found at %s', dataset_path);
    end

    opts = detectImportOptions(dataset_path, 'NumHeaderLines', 0);
    table_data = readtable(dataset_path, opts);

    feature_names = table_data.Properties.VariableNames(1:end-1);
    target_name = table_data.Properties.VariableNames{end};

    data = struct();
    data.X = table2array(table_data(:, feature_names));
    data.y = table2array(table_data(:, target_name));
    data.feature_names = feature_names;
    data.target_name = target_name;
end
