"""
训练预测脚本 - 神经网络的训练和预测 (Q5优化版)
目标函数: f(x1, x2) = 2*x1 + x1*x2, -1 < x1, x2 < 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from graph_module import node, Graph


def create_network_improved(hidden_size=5):

    g = Graph()
    
    # 创建输入节点
    n1 = node(typeOfNode="input", index=1, value=0.0)   # x1输入
    n2 = node(typeOfNode="input", index=2, value=0.0)   # x2输入
    
    init_scale = np.sqrt(2.0 / 2)  
    w_first_layer = []
    b_first_layer = []
    
    # 第一层权重 (输入到隐藏层)
    for i in range(hidden_size):
        w1 = node(typeOfNode="input", index=3+i*2, value=np.random.randn() * init_scale)
        w2 = node(typeOfNode="input", index=3+i*2+1, value=np.random.randn() * init_scale)
        w_first_layer.extend([w1, w2])
        b = node(typeOfNode="input", index=3+hidden_size*2+i, value=0.0)
        b_first_layer.append(b)
    
    # 第二层权重 (隐藏层到输出层)
    init_scale_out = np.sqrt(2.0 / hidden_size)
    v_layer = []
    for i in range(hidden_size):
        v = node(typeOfNode="input", index=3+hidden_size*2+hidden_size+i, 
                value=np.random.randn() * init_scale_out)
        v_layer.append(v)
    
    # 输出层偏置
    b_out = node(typeOfNode="input", index=3+hidden_size*3+hidden_size, value=0.0)
    
    # 初始化节点列表
    all_nodes = [n1, n2] + w_first_layer + b_first_layer + v_layer + [b_out]
    
    # 构建隐藏层计算节点
    hidden_outputs = []
    node_idx = 100
    
    for i in range(hidden_size):
        # 计算 x1*w1[i] + x2*w2[i] + b[i]
        mul1 = node(typeOfNode="mul", index=node_idx, inbound_nodes=[n1, w_first_layer[i*2]])
        all_nodes.append(mul1)
        node_idx += 1
        
        mul2 = node(typeOfNode="mul", index=node_idx, inbound_nodes=[n2, w_first_layer[i*2+1]])
        all_nodes.append(mul2)
        node_idx += 1
        
        add_sum = node(typeOfNode="add", index=node_idx, inbound_nodes=[mul1, mul2, b_first_layer[i]])
        all_nodes.append(add_sum)
        node_idx += 1
        
        relu = node(typeOfNode="ReLU", index=node_idx, inbound_nodes=[add_sum])
        all_nodes.append(relu)
        node_idx += 1
        
        hidden_outputs.append(relu)
    
    # 构建输出层
    output_muls = []
    for i in range(hidden_size):
        mul = node(typeOfNode="mul", index=node_idx, inbound_nodes=[hidden_outputs[i], v_layer[i]])
        all_nodes.append(mul)
        node_idx += 1
        output_muls.append(mul)
    
    # 最后的加法节点
    final_add = node(typeOfNode="add", index=node_idx, inbound_nodes=output_muls + [b_out])
    all_nodes.append(final_add)
    node_idx += 1
    
    output_node = node(typeOfNode="output", index=node_idx, inbound_nodes=[final_add])
    all_nodes.append(output_node)
    
    # 将所有节点添加到计算图中
    for node_instance in all_nodes:
        g.add_node(node_instance)
    
    # 返回计算图和参数节点列表
    param_nodes = w_first_layer + b_first_layer + v_layer + [b_out]
    input_nodes = [n1, n2]
    
    return g, param_nodes, input_nodes, output_node


def target_function(x1, x2):
    """目标函数: f(x1, x2) = 2*x1 + x1*x2"""
    return 2*x1 + x1*x2


def generate_training_data(num_samples=1000):
    """生成训练数据 - 均匀采样"""
    x1 = np.random.uniform(-1, 1, num_samples)
    x2 = np.random.uniform(-1, 1, num_samples)
    y = target_function(x1, x2)
    return x1, x2, y


def learning_rate_decay(initial_lr, epoch, decay_rate=0.95, decay_steps=100):
    """动态学习率调度 - 指数衰减"""
    return initial_lr * (decay_rate ** (epoch // decay_steps))


def train_network(g, param_nodes, input_nodes, output_node, x1_train, x2_train, y_train, 
                  learning_rate=0.1, epochs=2000, batch_size=64, 
                  early_stopping_patience=200, validation_split=0.2):

    losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 分割训练集和验证集
    n_val = int(len(x1_train) * validation_split)
    n_train = len(x1_train) - n_val
    
    indices = np.random.permutation(len(x1_train))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    x1_train_split = x1_train[train_indices]
    x2_train_split = x2_train[train_indices]
    y_train_split = y_train[train_indices]
    
    x1_val = x1_train[val_indices]
    x2_val = x2_train[val_indices]
    y_val = y_train[val_indices]
    
    print("=" * 60)
    print("开始优化的神经网络训练")
    print("=" * 60)
    print(f"训练集大小: {n_train}, 验证集大小: {n_val}")
    print(f"初始学习率: {learning_rate}, 批大小: {batch_size}")
    print(f"早停止耐心值: {early_stopping_patience}")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 计算动态学习率
        current_lr = learning_rate_decay(learning_rate, epoch, decay_rate=0.95, decay_steps=200)
        
        total_loss = 0
        num_batches = 0
        
        # 数据打乱
        shuffled_indices = np.random.permutation(n_train)
        
        # 批量训练
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            batch_indices = shuffled_indices[batch_start:batch_end]
            current_batch_size = batch_end - batch_start
            
            # 初始化批梯度累加器
            batch_grads = {param: 0.0 for param in param_nodes}
            batch_loss = 0
            
            for idx in batch_indices:
                # 设置输入值
                input_nodes[0].value = x1_train_split[idx]
                input_nodes[1].value = x2_train_split[idx]
                
                # 前向传播
                prediction = g.forward()
                
                # 计算损失 (MSE)
                target = y_train_split[idx]
                loss = 0.5 * (prediction - target) ** 2
                batch_loss += loss
                
                # 设置输出节点的梯度为预测误差
                output_node.grad = prediction - target
                
                # 反向传播（计算梯度）
                g.backward()
                
                # 收集梯度到批累加器
                for param in param_nodes:
                    batch_grads[param] += param.grad
            
            # 批量参数更新 - 使用平均梯度
            for param in param_nodes:
                avg_grad = batch_grads[param] / current_batch_size
                param.value -= current_lr * avg_grad
            
            total_loss += batch_loss
            num_batches += 1
        
        # 计算平均训练损失
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # 计算验证损失
        val_loss = 0
        for x1, x2, y in zip(x1_val, x2_val, y_val):
            input_nodes[0].value = x1
            input_nodes[1].value = x2
            prediction = g.forward()
            val_loss += 0.5 * (prediction - y) ** 2
        val_loss /= len(x1_val)
        val_losses.append(val_loss)
        
        # 早停止检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | Patience: {patience_counter}/{early_stopping_patience}")
        
        # 早停止
        if patience_counter >= early_stopping_patience:
            print(f"\n早停止触发！验证损失在最后 {early_stopping_patience} 个epoch内未改进")
            print(f"最佳验证损失: {best_val_loss:.6f}")
            break
    
    print("=" * 60)
    print(f"训练完成！总epoch数: {epoch + 1}")
    print("=" * 60)
    
    return losses, val_losses


def evaluate_network(g, input_nodes, x1_test, x2_test):
    """评估神经网络在测试集上的表现"""
    predictions = []
    for x1, x2 in zip(x1_test, x2_test):
        input_nodes[0].value = x1
        input_nodes[1].value = x2
        prediction = g.forward()
        predictions.append(prediction)
    return np.array(predictions)


def main():
    print("Q5: 神经网络拟合函数 f(x1, x2) = 2*x1 + x1*x2")
    print("=" * 70 + "\n")
    
    # 生成训练数据
    print("【1】生成训练数据...")
    x1_train, x2_train, y_train = generate_training_data(2000)  # 增加数据量
    print(f"    数据范围: x1, x2 ∈ [-1, 1]")
    print(f"    样本数: {len(x1_train)}")
    print(f"    标签范围: [{y_train.min():.4f}, {y_train.max():.4f}]\n")
    
    # 创建改进的神经网络
    print("【2】划分训练集和验证集...")
    hidden_size = 5  # 增加隐藏层大小
    g, param_nodes, input_nodes, output_node = create_network_improved(hidden_size=hidden_size)
    
    # 训练神经网络
    print("【3】开始训练神经网络...")
    losses, val_losses = train_network(
        g, param_nodes, input_nodes, output_node,
        x1_train, x2_train, y_train,
        learning_rate=0.1,
        epochs=5000,
        batch_size=64,
        early_stopping_patience=500,
        validation_split=0.2
    )
    print()
    
    # 绘制损失曲线
    print("【4】绘制训练曲线...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(losses, label='Training Loss', linewidth=1.5, alpha=0.8)
    ax.plot(val_losses, label='Validation Loss', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Neural Network Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    print("    损失曲线已保存为 loss_curve.png")
    plt.close()
    
    # 生成测试网格
    print("\n【5】生成测试网格并评估...")
    x = np.arange(-1, 1.05, 0.1)
    y = np.arange(-1, 1.05, 0.1)
    X, Y = np.meshgrid(x, y)
    
    # 计算网络在测试网格上的预测
    Z_pred = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            input_nodes[0].value = X[i, j]
            input_nodes[1].value = Y[i, j]
            Z_pred[i, j] = g.forward()
    
    # 计算真实值
    Z_true = target_function(X, Y)
    
    # 计算评估指标
    mse = np.mean((Z_pred - Z_true) ** 2)
    mae = np.mean(np.abs(Z_pred - Z_true))
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(Z_pred - Z_true))
    
    print(f"    均方误差 (MSE):  {mse:.6f}")
    print(f"    平均绝对误差 (MAE): {mae:.6f}")
    print(f"    均方根误差 (RMSE): {rmse:.6f}")
    print(f"    最大误差:        {max_error:.6f}")
    
    # 保存预测结果到MAT文件
    savemat('prediction.mat', {'Z': Z_pred, 'X': X, 'Y': Y, 'Z_true': Z_true})
    print("    预测结果已保存到 prediction.mat")
    
    # 绘制结果对比
    print("\n【6】绘制结果对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 真实函数
    im1 = axes[0, 0].contourf(X, Y, Z_true, levels=20, cmap='viridis')
    axes[0, 0].set_title('True Function: $f(x_1, x_2) = 2x_1 + x_1x_2$', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('$x_1$')
    axes[0, 0].set_ylabel('$x_2$')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 网络预测
    im2 = axes[0, 1].contourf(X, Y, Z_pred, levels=20, cmap='viridis')
    axes[0, 1].set_title('Network Prediction', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('$x_1$')
    axes[0, 1].set_ylabel('$x_2$')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 误差分析
    error = Z_pred - Z_true
    im3 = axes[1, 0].contourf(X, Y, error, levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Prediction Error: Predicted - True', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('$x_1$')
    axes[1, 0].set_ylabel('$x_2$')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 绝对误差
    abs_error = np.abs(error)
    im4 = axes[1, 1].contourf(X, Y, abs_error, levels=20, cmap='YlOrRd')
    axes[1, 1].set_title('Absolute Error: |Predicted - True|', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('$x_1$')
    axes[1, 1].set_ylabel('$x_2$')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    print("    对比图已保存为 comparison.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("训练和评估完成！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
