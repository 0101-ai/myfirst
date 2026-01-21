import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DataProcessor:
    def __init__(self, model):
        """
        初始化数据处理器

        参数:
        model: MagneticFieldModel实例
        """
        self.model = model

    def prepare_magnetic_field_data(self, data_1mm, data_3mm, current=None):
        """
        准备磁场数据
        数据已经是nT单位，不需要转换

        参数:
        data_1mm: 1mm高度的测量数据，形状为(n_currents, 4)，包含电流和三个分量
        data_3mm: 3mm高度的测量数据，形状为(n_currents, 4)，包含电流和三个分量
        current: 选择特定电流值的数据，如果为None则使用所有电流值

        返回:
        processed_data: 处理后的磁场数据，形状为(n_scan_points, 3)
        """
        # 单位已经是nT，不需要转换
        data_1mm_nT = data_1mm.copy()
        data_3mm_nT = data_3mm.copy()

        # 如果指定了电流值，筛选该电流值的数据
        if current is not None:
            # 找到最接近指定电流值的数据行
            idx_1mm = np.argmin(np.abs(data_1mm_nT[:, 0] - current))
            idx_3mm = np.argmin(np.abs(data_3mm_nT[:, 0] - current))

            # 提取对应的磁场数据
            field_1mm = data_1mm_nT[idx_1mm, 1:4].reshape(1, 3)
            field_3mm = data_3mm_nT[idx_3mm, 1:4].reshape(1, 3)

            print(f"已选择电流值 {data_1mm_nT[idx_1mm, 0]}A (1mm) 和 {data_3mm_nT[idx_3mm, 0]}A (3mm) 的数据")
        else:
            # 使用所有电流值的数据
            field_1mm = data_1mm_nT[:, 1:4]
            field_3mm = data_3mm_nT[:, 1:4]
            print(f"使用所有电流值的数据，共 {len(field_1mm)} 组")

        # 根据扫描平面的设置，调整磁场数据的形状
        # 在这个简化示例中，我们假设每个扫描平面中所有点测量到的磁场值相同
        # 真实情况下，应该根据实际测量数据调整
        n_points_per_plane = self.model.n_scan_points // 2  # 每个平面的点数

        # 扩展数据到所有扫描点
        if current is not None:
            # 单个电流值
            field_1mm_expanded = np.tile(field_1mm, (n_points_per_plane, 1))
            field_3mm_expanded = np.tile(field_3mm, (n_points_per_plane, 1))
        else:
            # 所有电流值，取平均
            field_1mm_avg = np.mean(field_1mm, axis=0).reshape(1, 3)
            field_3mm_avg = np.mean(field_3mm, axis=0).reshape(1, 3)
            field_1mm_expanded = np.tile(field_1mm_avg, (n_points_per_plane, 1))
            field_3mm_expanded = np.tile(field_3mm_avg, (n_points_per_plane, 1))

        # 合并两个平面的数据
        processed_data = np.vstack([field_1mm_expanded, field_3mm_expanded])

        print(f"已准备磁场数据，形状为{processed_data.shape}")
        return processed_data

    def generate_test_plane(self, height, x_range=(-10, 110), y_range=(-30, 30), nx=6, ny=4):
        """
        生成指定高度的测试平面点

        参数:
        height: 测试平面的高度 (mm)
        x_range: x方向范围 (mm)
        y_range: y方向范围 (mm)
        nx: x方向的点数
        ny: y方向的点数

        返回:
        test_points: 测试点坐标，形状为(nx*ny, 3)
        """
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        test_points = np.zeros((nx * ny, 3))
        test_points[:, 0] = X.flatten()
        test_points[:, 1] = Y.flatten()
        test_points[:, 2] = height

        return test_points

    def prepare_training_data(self):
        """
        准备神经网络训练数据

        返回:
        X_train, X_test, y_train, y_test: 训练集和测试集
        """
        if self.model.measured_fields is None:
            raise ValueError("请先加载测量数据")

        # 计算格林函数矩阵
        G = self.model.compute_greens_function_matrix()

        # 使用StandardScaler标准化输入和输出
        X = self.model.scaler_X.fit_transform(G)
        y = self.model.scaler_y.fit_transform(self.model.measured_fields.reshape(-1, 1)).flatten()

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 确保形状正确
        print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")

        return X_train, X_test, y_train, y_test

    def build_neural_network(self, hidden_layers=[32, 16], learning_rate=0.001):
        """
        构建神经网络模型

        参数:
        hidden_layers: 隐藏层神经元数量列表
        learning_rate: 学习率

        返回:
        model: 构建的神经网络模型
        """
        model = Sequential()

        # 输入层 - 注意明确指定输入形状
        input_dim = 2 * self.model.n_dipoles
        model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))

        # 隐藏层
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))

        # 输出层 - 一个值
        model.add(Dense(1))

        # 编译模型
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        # 打印模型摘要
        model.summary()

        self.model.model = model
        print(f"已构建神经网络模型：{hidden_layers}")
        return model

    def train_model(self, epochs=20, batch_size=8, validation_split=0.2, verbose=1):
        """
        训练神经网络模型

        参数:
        epochs: 训练轮数
        batch_size: 批量大小
        validation_split: 验证集比例
        verbose: 显示详细程度

        返回:
        history: 训练历史
        """
        if self.model.model is None:
            self.build_neural_network()

        X_train, X_test, y_train, y_test = self.prepare_training_data()

        # 确保y_train的形状正确（必须是二维的，但每个样本只有一个输出值）
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)

        print(f"训练前确认形状: X_train={X_train.shape}, y_train={y_train.shape}")

        history = self.model.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

        # 评估模型
        loss = self.model.model.evaluate(X_test, y_test, verbose=0)
        print(f"测试集损失: {loss:.6f}")

        return history