# 数据说明

## 文件结构
- `raw/diabetes.csv`：原始糖尿病数据集，共 442 条样本（来自 scikit-learn 数据集），最后一列为目标变量。
- `processed/diabetes_cleaned.mat`：运行 `main_analysis` 或 `data_preprocessor` 后生成的清洗结果，包含 `X`、`y` 以及特征名称。
- `processed/train_test_split.mat`：保存标准化及划分后的 `split` 结构，便于快速复现实验。

## 特征说明
常见特征包括：年龄、性别、BMI、血压以及 6 个血清指标。所有输入在模型训练前都会执行零均值、单位方差标准化。

## 使用建议
1. 不要直接修改 `processed/` 中的 `.mat` 文件，它们会在重新运行流程时被覆盖。
2. 若需替换为其它数据集，请保持 CSV 最后一列为目标变量并在表头提供特征名。
3. 对外部数据执行额外清洗后，可手动放入 `processed/` 目录，供 `result_visualization` 或自定义脚本加载。
