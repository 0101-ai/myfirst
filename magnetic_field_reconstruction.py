import numpy as np
import matplotlib.pyplot as plt
from model import MagneticFieldModel
from data_processor import DataProcessor
from visualizer import MagneticFieldVisualizer


def main():
    """主函数，使用实际测量数据进行磁场重构"""

    # 1. 导入实际测量数据 (单位已为nT)
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

    # 2. 创建模型、数据处理器和可视化器
    model = MagneticFieldModel(wire_length=100)  # 长度单位：mm
    processor = DataProcessor(model)
    visualizer = MagneticFieldVisualizer(model)

    # 3. 设置偶极子平面
    model.setup_dipole_plane(x_range=(0, 100), y_range=(-20, 20), z_value=0, nx=5, ny=3)

    # 4. 设置扫描平面
    model.setup_scan_planes(heights=[1, 3], x_range=(-10, 110), y_range=(-30, 30), nx=6, ny=4)

    # 5. 选择一个电流值的数据进行分析
    selected_current = 3.0  # 选择3.0A的数据

    # 6. 准备磁场数据
    field_data = processor.prepare_magnetic_field_data(data_1mm, data_3mm, current=selected_current)

    # 7. 加载测量数据
    model.load_measured_data(field_data)

    # 8. 构建神经网络
    processor.build_neural_network(hidden_layers=[32, 16])

    # 9. 训练模型
    print("开始训练神经网络模型...")
    history = processor.train_model(epochs=30, batch_size=8, verbose=1)

    # 10. 可视化训练历史
    visualizer.plot_training_history(history)

    # 11. 预测偶极子磁矩
    print("预测偶极子磁矩...")
    magnetic_moments = model.predict_magnetic_moments()

    # 12. 可视化偶极子磁矩分布
    visualizer.visualize_dipole_moments(magnetic_moments)

    # 13. 预测不同高度的磁场
    heights = [1, 3, 5, 10]  # 1mm, 3mm, 5mm, 10mm

    print("\n预测不同高度的磁场分布...")
    for height in heights:
        print(f"\n高度: {height}mm")

        # 生成测试点
        test_points = processor.generate_test_plane(height=height)

        # 预测磁场
        predicted_field = model.predict_magnetic_field(test_points)

        # 计算平均磁场强度
        avg_field = np.mean(np.sqrt(np.sum(predicted_field ** 2, axis=1)))
        print(f"平均磁场强度: {avg_field:.2f} nT")

        # 可视化预测结果
        for component in range(3):
            component_names = ['Bx', 'By', 'Bz']
            visualizer.visualize_magnetic_field(
                height, None, predicted_field,
                field_component=component,
                title=f'预测的{component_names[component]}分量 ({height}mm高度)'
            )

    # 14. 比较不同电流值下的预测
    print("\n比较不同电流值下的预测...")
    visualizer.compare_currents(data_1mm, data_3mm, currents=[1.0, 2.0, 3.0])

    print("\n磁场重构完成！")


if __name__ == "__main__":
    main()