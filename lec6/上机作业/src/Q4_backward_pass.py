"""
反向传播脚本 - 演示计算图的反向传播
"""

from graph_module import node, Graph


def backward_pass_example():
    """演示反向传播的示例"""
    print("=" * 50)
    print("反向传播演示")
    print("=" * 50)
    
    # 创建计算图
    g = Graph()

    # 创建输入节点
    n1 = node(typeOfNode="input", index=1, value=2.0)   # x1输入
    n2 = node(typeOfNode="input", index=2, value=-2.0)  # x2输入
    n3 = node(typeOfNode="input", index=3, value=1.0)   # w11输入
    n4 = node(typeOfNode="input", index=4, value=1.0)   # w21输入
    n5 = node(typeOfNode="input", index=5, value=1.0)   # w12输入
    n6 = node(typeOfNode="input", index=6, value=1.0)   # w22输入
    n7 = node(typeOfNode="input", index=7, value=1.0)   # w13输入
    n8 = node(typeOfNode="input", index=8, value=1.0)   # w23输入
    n9 = node(typeOfNode="input", index=9, value=0.0)   # b1输入
    n12 = node(typeOfNode="input", index=12, value=0.0) # b2输入
    n15 = node(typeOfNode="input", index=15, value=0.0) # b3输入
    n21 = node(typeOfNode="input", index=21, value=1.0) # v1输入
    n23 = node(typeOfNode="input", index=22, value=1.0) # v2输入
    n25 = node(typeOfNode="input", index=23, value=1.0) # v3输入
    n30 = node(typeOfNode="input", index=30, value=0.0) # b4输入

    # 构建隐藏层第一个神经元的计算节点
    n10 = node(typeOfNode="mul", index=10, inbound_nodes=[n1, n3])  # x1 * w11
    n11 = node(typeOfNode="mul", index=11, inbound_nodes=[n2, n4])  # x2 * w21
    n13 = node(typeOfNode="mul", index=13, inbound_nodes=[n1, n5])  # x1 * w12
    n14 = node(typeOfNode="mul", index=14, inbound_nodes=[n2, n6])  # x2 * w22
    n16 = node(typeOfNode="mul", index=16, inbound_nodes=[n1, n7])  # x1 * w13
    n17 = node(typeOfNode="mul", index=17, inbound_nodes=[n2, n8])  # x2 * w23
    
    n18 = node(typeOfNode="add", index=18, inbound_nodes=[n10, n11, n9])  # x1*w11 + x2*w21 + b1
    n19 = node(typeOfNode="add", index=19, inbound_nodes=[n13, n14, n12])  # x1*w12 + x2*w22 + b2
    n20 = node(typeOfNode="add", index=20, inbound_nodes=[n16, n17, n15])  # x1*w13 + x2*w23 + b3 
    
    n22 = node(typeOfNode="ReLU", index=22, inbound_nodes=[n18])  # ReLU(x1*w11 + x2*w21 + b1)
    n24 = node(typeOfNode="ReLU", index=24, inbound_nodes=[n19])  # ReLU(x1*w12 + x2*w22 + b2)
    n26 = node(typeOfNode="ReLU", index=26, inbound_nodes=[n20])  # ReLU(x1*w13 + x2*w23 + b3)
    
    n27 = node(typeOfNode="mul", index=27, inbound_nodes=[n22, n21])  # v1 * ReLU(...)
    n28 = node(typeOfNode="mul", index=28, inbound_nodes=[n24, n23])  # v2 * ReLU(...)
    n29 = node(typeOfNode="mul", index=29, inbound_nodes=[n26, n25])  # v3 * ReLU(...)
    
    n31 = node(typeOfNode="add", index=31, inbound_nodes=[n27, n28, n29, n30])  # v1*ReLU(...) + v2*ReLU(...) + v3*ReLU(...) + b4
    n32 = node(typeOfNode="output", index=32, inbound_nodes=[n31])  # 输出节点
    
    # 将所有节点添加到计算图中
    all_nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15,
                 n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29,
                 n30, n31, n32]
    for node_instance in all_nodes:
        g.add_node(node_instance)
    
    # 执行前向传播计算
    output_value = g.forward()
    print(f"\n前向传播结果: {output_value}")
    
    # 执行反向传播计算梯度
    print("\n开始反向传播...")
    g.backward()
    print("反向传播完成！")
    
    # 获取关于x1和x2的梯度
    grad_x1 = n1.grad
    grad_x2 = n2.grad
    
    print(f"\n梯度结果:")
    print(f"∂y/∂x₁(2, -2) = {grad_x1}")
    print(f"∂y/∂x₂(2, -2) = {grad_x2}")
    
    print(f"\n参数梯度:")
    print(f"∂y/∂w₁₁ = {n3.grad}")
    print(f"∂y/∂w₂₁ = {n4.grad}")
    print(f"∂y/∂v₁ = {n21.grad}")
    print(f"∂y/∂v₂ = {n23.grad}")
    print(f"∂y/∂v₃ = {n25.grad}")



if __name__ == "__main__":
    backward_pass_example()
