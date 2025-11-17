# 用户指南

## 环境准备
- MATLAB R2022a 及以上（需要 Statistics and Machine Learning Toolbox）。
- 将 `project2` 根目录添加到 MATLAB 路径：
  ```matlab
  addpath(genpath('path/to/project2'));
  ```
- 确认 `data/raw/diabetes.csv` 存在，或根据需要替换成同格式数据。

## 快速开始
1. **运行主分析**
   ```matlab
   cd path/to/project2/src/scripts
   main_analysis
   ```
   输出：
   - `results/models/*`：训练好的模型
   - `results/reports/*`：性能指标与总结
   - `results/figures/*`：图表

2. **重新绘制图表**（无需重新训练）
   ```matlab
   result_visualization
   ```

3. **查看调参结果**
   - `results/reports/performance_metrics.txt`
   - `results/reports/feature_analysis.txt`
   - `results/reports/summary_report.md`

## 测试
- 单元测试位于 `src/tests/`。
- 在 MATLAB 命令行中逐个运行：
  ```matlab
  test_linear_reg;
  test_ridge_reg;
  test_lasso_reg;
  ```

## 常见问题
- **缺少数据**：确认 `data/raw/diabetes.csv` 存在。如果使用自定义路径，可在 `config/parameters.m` 中修改输出位置。
- **图表未生成**：确保在运行 `main_analysis` 后再执行 `result_visualization`，并检查 `results/reports/tuning_summary.mat` 是否存在。
- **无 Statistics Toolbox**：`lasso` 函数属于该工具箱，未安装会导致错误。
