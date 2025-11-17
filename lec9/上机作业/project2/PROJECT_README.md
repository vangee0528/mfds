# Project2 · Diabetes Regression Benchmark

本项目实现了普通线性回归、岭回归与 Lasso 在糖尿病数据集上的对比实验，覆盖数据处理、调参、评估与可视化全流程。

## 快速开始
1. 确认 `data/raw/diabetes.csv` 存在。
2. 在 MATLAB 中添加项目路径：
   ```matlab
   addpath(genpath('path/to/project2'));
   ```
3. 运行主脚本：
   ```matlab
   main_analysis
   ```
4. 查看 `results/` 目录下的模型、指标与图表。若只需重新绘图，运行 `result_visualization`。

## 目录概览
- `config/`：路径与超参数配置。
- `data/`：原始与处理后的数据。
- `src/`：核心源码（模型、工具、脚本、测试）。
- `docs/`：理论背景、实现说明、用户指南。
- `results/`：输出的模型、图像与报告。

## 主要功能
- 统一的数据加载与标准化流程。
- k 折交叉验证寻找最优 $\lambda$。
- MSE / MAE / $R^2$ / 稀疏度对比。
- 系数路径、MSE 条形图与特征重要性可视化。
- 单元测试覆盖三个模型模块。

如需扩展到更多模型，可在 `src/models/` 添加实现，并在 `main_analysis` 中注册。
