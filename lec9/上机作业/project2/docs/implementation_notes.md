# 实现说明

## 目录与配置
- `config/paths.m`：集中管理路径，脚本无需手动拼接目录。
- `config/parameters.m`：包含随机种子、训练占比、标准化开关、$\lambda$ 网格以及输出文件路径。

## 数据流程
1. `data_loader.m` 读取 `data/raw/diabetes.csv` 并输出特征矩阵 `X`、标签 `y` 以及列名。
2. `data_preprocessor.m` 完成随机划分（70/30）、标准化以及 `.mat` 缓存。
3. 处理后的 `split` 结构被传入模型与调参脚本。

## 模型模块
- `linear_regression.m`：通过增广矩阵一次性解出 OLS 系数。
- `ridge_regression.m`：在增广矩阵上添加对角正则矩阵，忽略截距项的惩罚。
- `lasso_regression.m`：封装 MATLAB `lasso` 函数，关闭二次标准化以避免重复缩放。

## 脚本
- `main_analysis.m`：完整流水线，负责加载、调参、评估、保存模型与生成图表。
- `hyperparameter_tuning.m`：共享的 k 折交叉验证工具，输出最优 `lambda`、系数路径和最终模型。
- `result_visualization.m`：从 `results/reports/tuning_summary.mat` 与指标文本中恢复图表，无需重新训练。

## 结果输出
- `results/models/*.mat`：保存最佳模型权重，便于后续加载。
- `results/reports/*.txt/md`：包含性能指标、稀疏度统计与总结报告。
- `results/figures/*.png`：MSE 比较、系数路径、特征重要性图。

## 复现步骤
1. 确保 MATLAB 可访问 `project2` 根目录并将其添加到路径。
2. 运行 `src/scripts/main_analysis.m`。
3. 若仅需重新绘图，可运行 `src/scripts/result_visualization.m`。
4. 使用 `src/tests/test_*.m` 快速回归单元测试。
