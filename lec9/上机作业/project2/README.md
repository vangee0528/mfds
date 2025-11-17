参考项目结构：
```
project2/
│
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   │   └── diabetes.csv            # 原始糖尿病数据集
│   ├── processed/                  # 处理后的数据
│   │   ├── diabetes_cleaned.mat    # 清洗后的MATLAB数据
│   │   └── train_test_split.mat    # 训练测试集分割结果
│   └── README.md                   # 数据说明文档
│
├── src/                            # 源代码目录
│   ├── utils/                      # 工具函数
│   │   ├── data_loader.m           # 数据加载函数
│   │   ├── data_preprocessor.m     # 数据预处理函数
│   │   ├── evaluate_model.m        # 模型评估函数
│   │   └── plot_utilities.m        # 绘图工具函数
│   │
│   ├── models/                     # 模型实现
│   │   ├── linear_regression.m     # 普通线性回归
│   │   ├── ridge_regression.m      # 岭回归实现
│   │   ├── lasso_regression.m      # Lasso回归实现
│   │   └── lasso_reg.m             # 提供的Lasso对比函数
│   │
│   ├── scripts/                    # 主要执行脚本
│   │   ├── main_analysis.m         # 主分析脚本
│   │   ├── hyperparameter_tuning.m # 超参数调优
│   │   └── result_visualization.m  # 结果可视化
│   │
│   └── tests/                      # 测试脚本
│       ├── test_linear_reg.m       # 线性回归测试
│       ├── test_ridge_reg.m        # 岭回归测试
│       └── test_lasso_reg.m        # Lasso回归测试
│
├── results/                        # 结果输出目录
│   ├── figures/                    # 生成的图表
│   │   ├── coefficient_paths.png   # 系数路径图
│   │   ├── mse_comparison.png      # MSE比较图
│   │   ├── feature_importance.png  # 特征重要性图
│   │   └── performance_plots/      # 其他性能图表
│   │
│   ├── models/                     # 保存的模型
│   │   ├── best_ridge_model.mat    # 最优岭回归模型
│   │   ├── best_lasso_model.mat    # 最优Lasso模型
│   │   └── linear_model.mat        # 线性回归模型
│   │
│   └── reports/                    # 分析报告
│       ├── performance_metrics.txt # 性能指标
│       ├── feature_analysis.txt    # 特征分析结果
│       └── summary_report.md       # 总结报告
│
├── docs/                           # 文档目录
│   ├── theoretical_background.md   # 理论基础文档
│   ├── implementation_notes.md     # 实现说明
│   └── user_guide.md               # 用户指南
│
├── config/                         # 配置文件
│   ├── parameters.m                # 主要参数配置
│   └── paths.m                     # 路径配置
│
└── PROJECT_README.md               # 项目总说明文档
```