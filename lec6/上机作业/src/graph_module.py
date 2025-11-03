"""
计算图模块 - 定义节点和计算图的基础类
包含：node 类和 Graph 类
"""


class node:
    def __init__(self, typeOfNode: str, index: int, value: float = 0.0, inbound_nodes=None):
        self.inbound_nodes = inbound_nodes if inbound_nodes is not None else []
        self.outbound_nodes = []
        self.typeOfNode = typeOfNode
        self.index = index
        self.value = value
        self.grad = 0.0  # 该节点的梯度
        
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        if self.typeOfNode == "input":
            pass
        elif self.typeOfNode == "output":
            self.value = self.inbound_nodes[0].value
        elif self.typeOfNode == "add":
            total = 0
            for n in self.inbound_nodes:
                total += n.value
            self.value = total
        elif self.typeOfNode == "mul":
            product = 1
            for n in self.inbound_nodes:
                product *= n.value
            self.value = product
        elif self.typeOfNode == "ReLU":
            input_value = self.inbound_nodes[0].value
            self.value = max(0, input_value)
        else:
            raise ValueError(f"Unsupported node type: {self.typeOfNode}")

    def backward(self):
        # 计算该节点关于其输入节点的梯度
        if self.typeOfNode == "output":
            # 输出节点的梯度由外部设置（不需要在这里处理）
            # 只需将梯度传播给输入节点
            for n in self.inbound_nodes:
                n.grad += self.grad
            
        elif self.typeOfNode == "add":
            # 加法节点的梯度均匀分配到所有输入
            for n in self.inbound_nodes:
                n.grad += self.grad
                
        elif self.typeOfNode == "mul":
            # 乘法节点的梯度需要乘以其他输入的值
            for i, n in enumerate(self.inbound_nodes):
                product = 1.0
                for j, other_n in enumerate(self.inbound_nodes):
                    if i != j:
                        product *= other_n.value
                n.grad += self.grad * product
                
        elif self.typeOfNode == "ReLU":
            # ReLU节点的梯度：输入>0时为1，否则为0
            input_value = self.inbound_nodes[0].value
            if input_value > 0:
                self.inbound_nodes[0].grad += self.grad * 1.0
            else:
                self.inbound_nodes[0].grad += self.grad * 0.0


class Graph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node: node):
        self.nodes.append(node)

    def topological_sort(self) -> list:
        # 使用Kahn算法进行拓扑排序
        in_degree = {}
        for n in self.nodes:
            in_degree[n] = 0
        
        for n in self.nodes:
            for out_node in n.outbound_nodes:
                in_degree[out_node] += 1
        
        queue = [n for n in self.nodes if in_degree[n] == 0]
        sorted_nodes = []
        
        while queue:
            n = queue.pop(0)
            sorted_nodes.append(n)
            
            for out_node in n.outbound_nodes:
                in_degree[out_node] -= 1
                if in_degree[out_node] == 0:
                    queue.append(out_node)
        
        return sorted_nodes
    
    def forward(self):
        sorted_nodes = self.topological_sort()
        for n in sorted_nodes:
            n.forward()
        return sorted_nodes[-1].value
    
    def backward(self):
        # 保存输出节点的梯度（由外部设置，表示损失对输出的梯度）
        output_node = self.nodes[-1]
        saved_output_grad = output_node.grad
        
        # 初始化所有节点的梯度为0
        for n in self.nodes:
            n.grad = 0.0
        
        # 恢复输出节点的梯度
        output_node.grad = saved_output_grad
            
        # 反向传播计算梯度
        sorted_nodes = self.topological_sort()
        reverse_sorted = sorted_nodes[::-1]  # 逆序
        
        # 反向传播（从输出到输入）
        for node in reverse_sorted:
            node.backward()
