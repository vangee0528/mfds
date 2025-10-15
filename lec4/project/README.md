# 二维 Sinc 神经网络拟合项目

## 快速开始
1. 打开 MATLAB，切换到 `project` 目录。
2. 运行
   ```matlab
   addpath(fullfile(pwd, 'src'));
   main
   ```
3. 训练结束后，结果保存在 `output/figures/` 中，同时命令行显示测试集平均相对误差。

## 目录说明
- `src/`：自实现的前馈网络训练与可视化代码。
- `data/`：预留数据目录（当前训练数据在运行时随机生成）。
- `output/figures/`：脚本自动导出的图像与 `sinc_nn_results.txt`。
- `docs/`：实验报告及说明文档。

## 依赖
- MATLAB R2020a 及以上版本。
- 无需 Neural Network Toolbox，所有训练逻辑均为自定义实现。
