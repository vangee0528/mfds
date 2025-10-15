# 二维 Sinc 函数拟合实验总结

## 实验目标
使用自编前馈神经网络逼近二维 sinc 函数，比较预测与真值的误差。

## 方法概述
- 样本在 $[-8,8]^2$ 中均匀采样，避开原点 $10^{-6}$ 邻域。
- 网络结构：输入 2 → 隐藏层 20 → 隐藏层 20 → 输出 1，隐藏层激活为 `tanh`。
- 训练：批大小 256，最大 800 轮，学习率 0.01，Xavier 初始化，均方误差损失。
- 反向传播与权重更新全部自行实现。

## 运行步骤
```matlab
addpath(fullfile(pwd, 'src'));
main
```
训练完成后，`output/figures/` 会生成：
- `sinc_true_surface.png`
- `sinc_pred_surface.png`
- `sinc_abs_error_surface.png`
- `training_performance.png`
- `sinc_nn_results.txt`

## 结果概览
- 训练/验证 MSE 随迭代稳定下降，无过拟合迹象。
- 测试集平均相对误差约为 $10^{-3}$ 量级（具体数值见 `sinc_nn_results.txt`）。
- 预测曲面与真值高度一致，误差主要集中在原点附近。

## 后续工作
- 尝试自适应学习率（Adam、RMSProp）。
- 引入非均匀采样以增强原点附近的精度。
- 扩展为多种基函数逼近的对比实验。
