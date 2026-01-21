import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MagneticFieldVisualizer:
    def __init__(self, model):
        """
        初始化可视化器

        参数:
        model: MagneticFieldModel实例
        """
        self.model = model

    def plot_training_history(self, history):
        """
        绘制训练历史

        参数:
        history: 训练过程的历史数据
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型训练历史')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_magnetic_field(self, height, actual_field=None, predicted_field=None,
                                 field_component=1, title=None):
        """
        可视化磁场分布

        参数:
        height: 可视化的高度 (mm)
        actual_field: 实际磁场数据（可选）
        predicted_field: 预测磁场数据（可选）
        field_component: 要可视化的磁场分量（0:Bx, 1:By, 2:Bz）
        title: 图表标题
        """
        # 生成可视化平面上的网格点
        x_range = (-10, 110)  # mm
        y_range = (-30, 30)  # mm
        nx, ny = 6, 4

        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        # 生成测试点 (mm)
        test_points = np.zeros((nx * ny, 3))
        test_points[:, 0] = X.flatten()
        test_points[:, 1] = Y.flatten()
        test_points[:, 2] = height

        # 如果提供了预测磁场，则使用它
        if predicted_field is None and self.model.model is not None:
            predicted_field = self.model.predict_magnetic_field(test_points)

        component_names = ['Bx', 'By', 'Bz']
        component_name = component_names[field_component]

        plt.figure(figsize=(12, 5))

        # 绘制实际磁场（如果提供）
        if actual_field is not None:
            plt.subplot(1, 2, 1)
            actual_values = actual_field[:, field_component].reshape(ny, nx)
            im1 = plt.pcolormesh(X, Y, actual_values, cmap='viridis', shading='auto')
            plt.colorbar(im1, label=f'实际 {component_name} (nT)')
            plt.title(f'实际 {component_name} (z={height}mm)')
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')

            # 绘制导线位置
            plt.plot([0, self.model.wire_length], [0, 0], 'r-', linewidth=2, label='导线')
            plt.legend()

        # 绘制预测磁场（如果提供）
        if predicted_field is not None:
            plt_idx = 2 if actual_field is not None else 1
            plt.subplot(1, 2, plt_idx)
            predicted_values = predicted_field[:, field_component].reshape(ny, nx)
            im2 = plt.pcolormesh(X, Y, predicted_values, cmap='viridis', shading='auto')
            plt.colorbar(im2, label=f'预测 {component_name} (nT)')
            plt.title(f'预测 {component_name} (z={height}mm)')
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')

            # 绘制导线位置
            plt.plot([0, self.model.wire_length], [0, 0], 'r-', linewidth=2, label='导线')
            # 标记缺陷位置
            plt.axvline(x=self.model.defect_position, color='g', linestyle='--', linewidth=1.5, label='缺陷位置')
            plt.legend()

        if title:
            plt.suptitle(title)

        plt.tight_layout()
        plt.show()

    def visualize_dipole_moments(self, magnetic_moments=None):
        """
        可视化偶极子磁矩分布

        参数:
        magnetic_moments: 偶极子的磁矩，形状为(n_dipoles, 2)
        """
        if magnetic_moments is None:
            magnetic_moments = self.model.predict_magnetic_moments()

        # 提取偶极子位置
        x = self.model.dipoles[:, 0]
        y = self.model.dipoles[:, 1]

        # 提取磁矩分量
        mx = magnetic_moments[:, 0]
        my = magnetic_moments[:, 1]

        # 计算磁矩大小
        magnitude = np.sqrt(mx ** 2 + my ** 2)

        plt.figure(figsize=(10, 8))

        # 绘制偶极子位置和磁矩方向
        plt.quiver(x, y, mx, my, magnitude, cmap='jet', scale=20)
        plt.colorbar(label='磁矩大小')

        # 绘制导线位置
        plt.plot([0, self.model.wire_length], [0, 0], 'r-', linewidth=2, label='导线')
        # 标记缺陷位置
        plt.axvline(x=self.model.defect_position, color='g', linestyle='--', linewidth=1.5, label='缺陷位置')

        plt.title('磁偶极子矩分布')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_currents(self, data_1mm, data_3mm, currents=[1.0, 2.0, 3.0], height=3):
        """
        比较不同电流值下的磁场分布

        参数:
        data_1mm: 1mm高度的测量数据
        data_3mm: 3mm高度的测量数据
        currents: 要比较的电流值列表
        height: 测试平面高度 (mm)
        """
        from data_processor import DataProcessor
        processor = DataProcessor(self.model)

        plt.figure(figsize=(15, 5))

        for i, component in enumerate(['Bx', 'By', 'Bz']):
            plt.subplot(1, 3, i + 1)
            for current in currents:
                # 准备该电流值的数据
                field_data = processor.prepare_magnetic_field_data(data_1mm, data_3mm, current=current)
                self.model.load_measured_data(field_data)

                # 重新训练模型
                processor.build_neural_network(hidden_layers=[32, 16])
                processor.train_model(epochs=20, batch_size=8, verbose=0)

                # 预测中心线上的磁场
                x_values = np.linspace(0, 100, 100)  # mm
                center_points = np.zeros((100, 3))
                center_points[:, 0] = x_values
                center_points[:, 2] = height

                predicted_field = self.model.predict_magnetic_field(center_points)

                # 提取指定分量
                component_idx = ['Bx', 'By', 'Bz'].index(component)
                field_component = predicted_field[:, component_idx]

                # 绘制曲线
                plt.plot(x_values, field_component, label=f'{current}A')

            plt.title(f'{component}分量沿x轴 (y=0, z={height}mm)')
            plt.xlabel('x (mm)')
            plt.ylabel(f'{component} (nT)')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_3d_field(self, height, field_component=1):
        """
        以3D方式可视化磁场分布

        参数:
        height: 可视化的高度 (mm)
        field_component: 要可视化的磁场分量（0:Bx, 1:By, 2:Bz）
        """
        # 生成可视化平面上的网格点
        x_range = (-10, 110)  # mm
        y_range = (-30, 30)  # mm
        nx, ny = 20, 20  # 增加分辨率

        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        # 生成测试点 (mm)
        test_points = np.zeros((nx * ny, 3))
        test_points[:, 0] = X.flatten()
        test_points[:, 1] = Y.flatten()
        test_points[:, 2] = height

        # 预测磁场
        predicted_field = self.model.predict_magnetic_field(test_points)

        # 提取所需分量
        component_names = ['Bx', 'By', 'Bz']
        component_name = component_names[field_component]
        field_values = predicted_field[:, field_component].reshape(ny, nx)

        # 创建3D图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制3D曲面
        surf = ax.plot_surface(X, Y, field_values, cmap='viridis', edgecolor='none', alpha=0.8)

        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=f'{component_name} (nT)')

        # 添加说明
        ax.set_title(f'磁场{component_name}分量的3D分布 (z={height}mm)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel(f'{component_name} (nT)')

        # 标记导线位置
        x_wire = np.linspace(0, self.model.wire_length, 100)
        y_wire = np.zeros_like(x_wire)
        z_wire = np.zeros_like(x_wire) + np.min(field_values)
        ax.plot(x_wire, y_wire, z_wire, 'r-', linewidth=3, label='导线位置')

        # 标记缺陷位置
        ax.plot([self.model.defect_position], [0], [np.min(field_values)], 'go', markersize=10, label='缺陷位置')

        ax.legend()
        plt.tight_layout()
        plt.show()