import numpy as np
import matplotlib.pyplot as plt
from model import MagneticFieldModel
from data_processor import DataProcessor
from visualizer import MagneticFieldVisualizer
from config import WIRE_CONFIG, DIPOLE_CONFIG, SCAN_CONFIG, NEURAL_NETWORK_CONFIG


def load_test_data():
    """加载测试数据"""
    # 1mm高度的数据 [电流(A), X(nT), Y(nT), Z(nT), 合磁场(nT)]
    data_1mm = np.array([
        [0.1, 488.29, 206.37, 261.43, 618.54],
        [0.3, 1278.22, 530.97, 404.63, 1418.52],
        [0.5, 1961.82, 838.17, 1032.81, 2321.04],
        [0.9, 3408.61, 1472.56, 1688.39, 4053.01],
        [1.2, 4475.66, 1946.03, 2140.17, 5497.01],
        [1.5, 5528.10, 2428.05, 2769.22, 6810.55],
        [1.8, 6651.13, 2898.66, 3181.55, 8181.31],
        [2.1, 7729.68, 3372.75, 3753.35, 9547.96],
        [2.4, 8772.06, 3848.44, 4142.50, 10890.13],
        [2.7, 9908.07, 4329.12, 4730.46, 12378.60],
        [3.0, 10998.3, 4795.32, 5229.39, 13458.60],
        [3.2, 11872.5, 5142.66, 5329.50, 14510.11]
    ])

    # 3mm高度的数据 [电流(A), X(nT), Y(nT), Z(nT), 合磁场(nT)]
    data_3mm = np.array([
        [0.1, 113.04, 509.00, 71.01, 520.54],
        [0.3, 330.00, 1370.64, 770.71, 1604.51],
        [0.5, 568.01, 2241.07, 1264.63, 2612.96],
        [0.9, 1017.67, 4010.10, 1988.63, 4636.81],
        [1.2, 1369.25, 5346.81, 2408.53, 6063.44],
        [1.5, 1728.90, 6684.00, 3226.80, 7639.11],
        [1.8, 2070.83, 8001.78, 3606.25, 9042.37],
        [2.1, 2427.10, 9338.81, 4387.55, 10628.54],
        [2.4, 2784.55, 10666.00, 5072.88, 12132.85],
        [2.7, 3130.54, 11967.08, 5550.41, 13577.07],
        [3.0, 3487.75, 13288.81, 6391.10, 15136.33],
        [3.2, 3731.50, 14224.10, 6507.70, 16228.92]
    ])

    return data_1mm, data_3mm


def analyze_defect_effect():
    """分析缺陷对磁场的影响"""
    # 创建具有缺陷和无缺陷的两个模型
    model_with_defect = MagneticFieldModel(wire_length=WIRE_CONFIG['length'])
    model_no_defect = MagneticFieldModel(wire_length=WIRE_CONFIG['length'])

    # 移除无缺陷模型的缺陷效应
    # 通过修改defect_function方法，使其始终返回1（无扰动）
    def no_defect_function(self, x):
        return 1.0

    # 修改无缺陷模型的defect_function方法
    model_no_defect.defect_function = lambda x: 1.0

    # 设置模型参数
    for model in [model_with_defect, model_no_defect]:
        model.setup_dipole_plane(**DIPOLE_CONFIG)
        model.setup_scan_planes(**SCAN_CONFIG)

    # 加载测试数据
    data_1mm, data_3mm = load_test_data()

    # 创建数据处理器
    processor_with_defect = DataProcessor(model_with_defect)
    processor_no_defect = DataProcessor(model_no_defect)

    # 准备数据并训练模型
    current = 3.0
    field_data_with_defect = processor_with_defect.prepare_magnetic_field_data(data_1mm, data_3mm, current=current)
    field_data_no_defect = processor_no_defect.prepare_magnetic_field_data(data_1mm, data_3mm, current=current)

    model_with_defect.load_measured_data(field_data_with_defect)
    model_no_defect.load_measured_data(field_data_no_defect)

    processor_with_defect.build_neural_network(**NEURAL_NETWORK_CONFIG)
    processor_no_defect.build_neural_network(**NEURAL_NETWORK_CONFIG)

    processor_with_defect.train_model(epochs=NEURAL_NETWORK_CONFIG['epochs'], verbose=0)
    processor_no_defect.train_model(epochs=NEURAL_NETWORK_CONFIG['epochs'], verbose=0)

    # 创建可视化器
    visualizer_with_defect = MagneticFieldVisualizer(model_with_defect)
    visualizer_no_defect = MagneticFieldVisualizer(model_no_defect)

    # 比较缺陷和无缺陷模型的磁场分布
    height = 3  # mm

    # 生成测试点
    test_points = processor_with_defect.generate_test_plane(height=height)

    # 预测磁场
    field_with_defect = model_with_defect.predict_magnetic_field(test_points)
    field_no_defect = model_no_defect.predict_magnetic_field(test_points)

    # 计算差异场
    field_difference = field_with_defect - field_no_defect

    # 可视化比较
    for component_idx, component_name in enumerate(['Bx', 'By', 'Bz']):
        plt.figure(figsize=(15, 5))

        # 设置网格
        x_range = (-10, 110)
        y_range = (-30, 30)
        nx, ny = 6, 4
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        # 有缺陷模型
        plt.subplot(1, 3, 1)
        field_values = field_with_defect[:, component_idx].reshape(ny, nx)
        im1 = plt.pcolormesh(X, Y, field_values, cmap='viridis', shading='auto')
        plt.colorbar(im1, label=f'{component_name} (nT)')
        plt.title(f'有缺陷模型 - {component_name} (z={height}mm)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.plot([0, WIRE_CONFIG['length']], [0, 0], 'r-', linewidth=2, label='导线')
        plt.axvline(x=WIRE_CONFIG['defect_position'], color='g', linestyle='--', linewidth=1.5, label='缺陷位置')
        plt.legend()

        # 无缺陷模型
        plt.subplot(1, 3, 2)
        field_values = field_no_defect[:, component_idx].reshape(ny, nx)
        im2 = plt.pcolormesh(X, Y, field_values, cmap='viridis', shading='auto')
        plt.colorbar(im2, label=f'{component_name} (nT)')
        plt.title(f'无缺陷模型 - {component_name} (z={height}mm)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.plot([0, WIRE_CONFIG['length']], [0, 0], 'r-', linewidth=2, label='导线')
        plt.legend()

        # 差异场
        plt.subplot(1, 3, 3)
        field_values = field_difference[:, component_idx].reshape(ny, nx)
        im3 = plt.pcolormesh(X, Y, field_values, cmap='RdBu_r', shading='auto')
        plt.colorbar(im3, label=f'差异 {component_name} (nT)')
        plt.title(f'缺陷引起的差异场 - {component_name} (z={height}mm)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.plot([0, WIRE_CONFIG['length']], [0, 0], 'r-', linewidth=2, label='导线')
        plt.axvline(x=WIRE_CONFIG['defect_position'], color='g', linestyle='--', linewidth=1.5, label='缺陷位置')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # 绘制中心线上的磁场变化
    plt.figure(figsize=(15, 5))

    # 生成中心线上的点
    x_values = np.linspace(0, WIRE_CONFIG['length'], 100)
    center_points = np.zeros((100, 3))
    center_points[:, 0] = x_values
    center_points[:, 2] = height

    # 预测中心线上的磁场
    center_field_with_defect = model_with_defect.predict_magnetic_field(center_points)
    center_field_no_defect = model_no_defect.predict_magnetic_field(center_points)
    center_field_diff = center_field_with_defect - center_field_no_defect

    # 绘制三个分量
    for i, component in enumerate(['Bx', 'By', 'Bz']):
        plt.subplot(1, 3, i + 1)
        plt.plot(x_values, center_field_with_defect[:, i], 'b-', label='有缺陷')
        plt.plot(x_values, center_field_no_defect[:, i], 'r--', label='无缺陷')
        plt.plot(x_values, center_field_diff[:, i], 'g-.', label='差异')

        plt.axvline(x=WIRE_CONFIG['defect_position'], color='k', linestyle=':', linewidth=1.5, label='缺陷位置')

        plt.title(f'中心线上{component}分量 (z={height}mm)')
        plt.xlabel('x (mm)')
        plt.ylabel(f'{component} (nT)')
        plt.grid(True)
        plt.legend()

    plt.tight_layout