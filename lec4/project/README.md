# 二维 Sinc 神经网络拟合项目（Python 版）

## 快速开始
1. 在终端切换到 `project` 目录并创建虚拟环境（可选）。
2. 运行主程序：
   ```bash
   python -m src.main
   ```
4. 训练结束后，结果保存在 `output/figures/` 中，同时命令行显示测试集平均相对误差。

## 目录说明
- `src/`：自实现的前馈网络训练、可视化与入口脚本。
- `data/`：预留数据目录（当前训练数据由脚本在线生成）。
- `output/figures/`：自动导出的图像与 `sinc_nn_results.txt` 指标文件。
- `docs/`：实验报告及补充说明。

## 依赖
- Python 3.10 及以上版本。

