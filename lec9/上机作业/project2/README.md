# Project 2 · 回归方法比较

本项目在 `project2/` 目录下给出了完整的 MATLAB 实验代码，用于比较普通线性回归（OLS）、岭回归（Ridge）与 Lasso 回归在 `diabetes.csv` 数据集上的表现。

## 目录结构
```
project2/
├── data/                # 数据说明（如有需要可将 CSV 拷贝到此处）
├── results/             # 运行后生成的表格与图像
│   └── figures/         # 系数路径、性能对比图
├── src/                 # 所有函数脚本
│   ├── compute_mse.m
│   ├── fit_lasso_regression.m
│   ├── fit_linear_regression.m
│   ├── fit_ridge_regression.m
│   ├── plot_coeff_paths.m
│   ├── plot_model_performance.m
│   ├── predict_regression.m
│   └── prepare_diabetes_data.m
└── main.m               # 主脚本，一键完成实验流程
```

## 运行方式
1. 打开 MATLAB，将当前工作目录切换到 `project2/`。
2. 确保 `lec9/上机作业/diabetes.csv` 存在；如需更改路径，可在 `main.m` 中修改 `dataPath`。
3. 在命令行运行：
   ```matlab
   main
   ```
4. 程序会完成以下步骤：
   - 读取并标准化特征，按 7:3 随机划分训练/测试集；
   - 训练 OLS、Ridge（带 5 折交叉验证）和 Lasso（自带 CV）；
   - 计算测试集 MSE、统计非零特征数；
   - 绘制系数路径图、性能对比图；
   - 输出 `results/performance_report.txt` 与 `results/performance_metrics.csv`。

## 输出说明
- `results/performance_metrics.csv`：包含每种模型的测试 MSE、非零系数数量、最优正则化系数。
- `results/performance_report.txt`：文字总结（包括超参数列表和最优 λ）。
- `results/figures/coefficient_paths.png`：参考 `eg` 目录作图风格的系数路径对比。
- `results/figures/performance_comparison.png`：MSE 与稀疏性柱状图，对比三种方法。

## 讨论建议
运行完成后，可结合图表讨论：
- Ridge 随 λ 增大系数收缩但不为零，适合缓解多重共线；
- Lasso 会产生稀疏解，便于自动特征选择；
- OLS 没有正则项，可能对噪声敏感。依据测试 MSE 及特征数量，选择更适合该数据集的模型并说明原因。
