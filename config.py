# 磁场重构系统配置参数

# 导线参数
WIRE_CONFIG = {
    'length': 100,        # 导线长度 (mm)
    'width': 3.0,         # 导线宽度 (mm)
    'thickness': 0.035,   # 导线厚度 (mm)
    'defect_position': 25 # 缺陷位置 (mm)
}

# 偶极子平面配置
DIPOLE_CONFIG = {
    'x_range': (0, 100),   # x范围 (mm)
    'y_range': (-20, 20),  # y范围 (mm)
    'z_value': 0,          # z位置 (mm)
    'nx': 5,               # x方向偶极子数量
    'ny': 3                # y方向偶极子数量
}

# 扫描平面配置
SCAN_CONFIG = {
    'heights': [1, 3],      # 高度列表 (mm)
    'x_range': (-10, 110),  # x范围 (mm)
    'y_range': (-30, 30),   # y范围 (mm)
    'nx': 6,                # x方向扫描点数量
    'ny': 4                 # y方向扫描点数量
}

# 预测设置
PREDICTION_CONFIG = {
    'heights': [1, 3, 5, 10],  # 要预测的高度 (mm)
    'currents': [1.0, 2.0, 3.0] # 要比较的电流值 (A)
}

# 神经网络配置
NEURAL_NETWORK_CONFIG = {
    'hidden_layers': [32, 16],  # 隐藏层神经元数量
    'learning_rate': 0.001,     # 学习率
    'epochs': 40,               # 训练轮数
    'batch_size': 8,            # 批量大小
    'validation_split': 0.2     # 验证集比例
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'components': ['Bx', 'By', 'Bz'],  # 磁场分量
    'component_indices': {'Bx': 0, 'By': 1, 'Bz': 2}  # 索引映射
}