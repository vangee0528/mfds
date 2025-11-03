# 第六次上机作业

## 1.  写出上面网络的公式
\[y_1 = \sum_{j=1}^3 v_j \cdot \text{ReLU}\left(\sum_{i=1}^2 w_{ij}x_i + b_j\right) + b_4\]

## 2. 绘制计算图如下
![计算图](FIGURES/computation_graph.png)

## 3. 计算前向传播
按照上述计算图编写Python代码见`src/Q3_forward_pass.py`：

运行计算，得到结果为0.0

## 4. 反向传播计算梯度

按照上述计算图编写Python代码见`src/Q4_backward_pass.py`：

运行计算，得到各参数梯度如下：

梯度结果:
\(\frac{\partial y}{\partial x_1}(2, -2) = 0.0\)
\(\frac{\partial y}{\partial x_2}(2, -2) = 0.0\)

## 5. 训练网络预测函数\(f(x_1, x_2) = 2x_1 + x_1x_2\)

按照上述计算图编写训练代码,见`src/Q5_train_predict.py`。

矩阵保存在data/3220102895.mat中。

经过2160次迭代训练后触发早停，最终损失值约为0.000710.


### 训练损失曲线
![训练损失曲线](FIGURES/loss_curve.png)
### 预测结果对比图
![预测结果对比图](FIGURES/comparison.png)


### 训练日志
```console
Q5: 神经网络拟合函数 f(x1, x2) = 2*x1 + x1*x2
======================================================================

【1】生成训练数据...
    数据范围: x1, x2 ∈ [-1, 1]
    样本数: 2000
    标签范围: [-2.8083, 2.8436]

【2】划分训练集和验证集...
【3】开始训练神经网络...
============================================================
开始优化的神经网络训练
============================================================
训练集大小: 1600, 验证集大小: 400
初始学习率: 0.1, 批大小: 64
早停止耐心值: 500
============================================================
Epoch    0 | Train Loss: 26.996739 | Val Loss: 0.259363 | LR: 0.100000 | Patience: 0/500
Epoch  100 | Train Loss: 0.215769 | Val Loss: 0.003358 | LR: 0.100000 | Patience: 0/500
Epoch  200 | Train Loss: 0.113023 | Val Loss: 0.001815 | LR: 0.095000 | Patience: 0/500
Epoch  300 | Train Loss: 0.081288 | Val Loss: 0.001334 | LR: 0.095000 | Patience: 1/500
Epoch  400 | Train Loss: 0.066605 | Val Loss: 0.001118 | LR: 0.090250 | Patience: 0/500
Epoch  500 | Train Loss: 0.058881 | Val Loss: 0.001049 | LR: 0.090250 | Patience: 0/500
Epoch  600 | Train Loss: 0.056766 | Val Loss: 0.000995 | LR: 0.085737 | Patience: 5/500
Epoch  700 | Train Loss: 0.055025 | Val Loss: 0.000950 | LR: 0.085737 | Patience: 0/500
Epoch  800 | Train Loss: 0.053390 | Val Loss: 0.000931 | LR: 0.081451 | Patience: 8/500
Epoch  900 | Train Loss: 0.052168 | Val Loss: 0.000903 | LR: 0.081451 | Patience: 14/500
Epoch 1000 | Train Loss: 0.050349 | Val Loss: 0.000869 | LR: 0.077378 | Patience: 0/500
Epoch 1100 | Train Loss: 0.048141 | Val Loss: 0.000898 | LR: 0.077378 | Patience: 4/500
Epoch 1200 | Train Loss: 0.045110 | Val Loss: 0.000782 | LR: 0.073509 | Patience: 5/500
Epoch 1300 | Train Loss: 0.042686 | Val Loss: 0.000754 | LR: 0.073509 | Patience: 7/500
Epoch 1400 | Train Loss: 0.041821 | Val Loss: 0.000741 | LR: 0.069834 | Patience: 7/500
Epoch 1500 | Train Loss: 0.041178 | Val Loss: 0.000722 | LR: 0.069834 | Patience: 7/500
Epoch 1600 | Train Loss: 0.040706 | Val Loss: 0.000718 | LR: 0.066342 | Patience: 21/500
Epoch 1700 | Train Loss: 0.040647 | Val Loss: 0.000723 | LR: 0.066342 | Patience: 41/500
Epoch 1800 | Train Loss: 0.040379 | Val Loss: 0.000723 | LR: 0.063025 | Patience: 141/500
Epoch 1900 | Train Loss: 0.040009 | Val Loss: 0.000724 | LR: 0.063025 | Patience: 241/500
Epoch 2000 | Train Loss: 0.039864 | Val Loss: 0.000721 | LR: 0.059874 | Patience: 341/500
Epoch 2100 | Train Loss: 0.039932 | Val Loss: 0.000721 | LR: 0.059874 | Patience: 441/500

早停止触发！验证损失在最后 500 个epoch内未改进
最佳验证损失: 0.000710
============================================================
训练完成！总epoch数: 2160
============================================================

【4】绘制训练曲线...
    损失曲线已保存为 loss_curve.png

【5】生成测试网格并评估...
    均方误差 (MSE):  0.001826
    平均绝对误差 (MAE): 0.031705
    均方根误差 (RMSE): 0.042727
    最大误差:        0.195061
    预测结果已保存到 prediction.mat

【6】绘制结果对比图...
    对比图已保存为 comparison.png

======================================================================
训练和评估完成！
======================================================================
```







