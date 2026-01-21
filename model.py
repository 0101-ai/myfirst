import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MagneticFieldModel:
    def __init__(self, wire_length=100, wire_position=None):
        """
        初始化磁场重构模型类

        参数:
        wire_length: 导线长度(mm)，默认100mm
        wire_position: 导线位置，默认沿着x轴从0到wire_length
        """
        self.wire_length = wire_length  # 单位: mm

        if wire_position is None:
            self.wire_position = np.array([[0, 0, 0], [wire_length, 0, 0]])
        else:
            self.wire_position = wire_position

        # 物理常数
        self.mu0 = 4 * np.pi * 1e-7  # 真空磁导率，单位H/m

        # 偶极子相关参数
        self.dipoles = None  # 偶极子位置
        self.n_dipoles = 0  # 偶极子数量

        # PCB导线特征
        self.width = 3.0  # 导线宽度，单位：mm
        self.thickness = 0.035  # 导线厚度，单位：mm
        self.defect_position = 25.0  # 缺陷位置，单位：mm

        # 扫描点相关参数
        self.scan_points = None  # 扫描点位置
        self.n_scan_points = 0  # 扫描点数量

        # 测量数据
        self.measured_fields = None

        # 神经网络模型
        self.model = None

        # 数据处理
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def setup_dipole_plane(self, x_range=(0, 100), y_range=(-20, 20), z_value=0,
                           nx=5, ny=3):
        """
        设置偶极子平面 (单位: mm)

        参数:
        x_range: x方向范围，默认为导线长度 (mm)
        y_range: y方向范围，默认为导线两侧各20mm (mm)
        z_value: 偶极子平面的z坐标值 (mm)
        nx: x方向的偶极子数量
        ny: y方向的偶极子数量
        """
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        # 创建偶极子位置数组 (单位: mm)
        dipoles = np.zeros((nx * ny, 3))
        dipoles[:, 0] = X.flatten()
        dipoles[:, 1] = Y.flatten()
        dipoles[:, 2] = z_value

        self.dipoles = dipoles
        self.n_dipoles = len(dipoles)

        print(f"已设置{self.n_dipoles}个偶极子在平面上")
        return self.dipoles

    def setup_scan_planes(self, heights=[1, 3], x_range=(-10, 110),
                          y_range=(-30, 30), nx=6, ny=4):
        """
        设置扫描平面 (单位: mm)

        参数:
        heights: 扫描平面的高度列表，例如[1mm, 3mm]
        x_range: x方向范围，默认比导线长度稍大 (mm)
        y_range: y方向范围，默认比偶极子平面稍大 (mm)
        nx: x方向的扫描点数量
        ny: y方向的扫描点数量
        """
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        # 创建所有扫描平面的扫描点 (单位: mm)
        scan_points = []
        for h in heights:
            plane_points = np.zeros((nx * ny, 3))
            plane_points[:, 0] = X.flatten()
            plane_points[:, 1] = Y.flatten()
            plane_points[:, 2] = h
            scan_points.append(plane_points)

        # 合并所有扫描平面的点
        self.scan_points = np.vstack(scan_points)
        self.n_scan_points = len(self.scan_points)

        print(f"已在{len(heights)}个平面上设置共{self.n_scan_points}个扫描点")
        return self.scan_points

    def defect_function(self, x):
        """
        定义缺陷特征函数 η(x)
        在缺陷位置处产生高斯型扰动

        参数:
            x: 导线上的位置坐标 (mm)
        返回:
            缺陷扰动系数
        """
        sigma = 1.0  # 高斯分布的标准差，控制缺陷特征宽度（1mm）
        # 返回基础值1.0加上一个高斯扰动（扰动幅度为0.1或10%）
        return 1.0 + 0.1 * np.exp(-((x - self.defect_position) ** 2) / (2 * sigma ** 2))

    def kernel_function(self, x, measure_point):
        """
        定义核函数，基于改进的物理模型

        参数:
            x: 导线上的点的位置 (mm)
            measure_point: 测量点的三维坐标 (x,y,z) (mm)
        返回:
            该点对测量点的磁场贡献系数 (用于计算nT)
        """
        # 计算导线上的点到测量点的相对位置分量 (mm)
        r_x = measure_point[0] - x  # x方向距离
        r_y = measure_point[1]  # y方向距离
        r_z = measure_point[2]  # z方向距离

        # 计算垂直于导线的距离分量 (mm)
        # 使用改进的方法，考虑r_z的权重
        r_perp = np.sqrt(r_y ** 2 + (0.24 * r_z) ** 2)

        # 防止除以零错误
        if r_perp < 1e-3:  # 小于0.001mm
            return 0

        # 计算总距离 (mm)
        r = np.sqrt(r_x ** 2 + r_perp ** 2)
        if r < 1e-3:  # 防止除以零错误
            return 0

        # 计算磁场核函数，依照1/r^(1.48)的关系（与垂直距离的关系）
        # 转换因子：1e9是从T转为nT，1e-3是从m转为mm
        conversion_factor = 1e9 * 1e-3  # T转nT，m转mm
        K = (self.mu0 / (4 * np.pi)) * (1 / np.power(r_perp * 1e-3, 1.48)) * conversion_factor

        return K

    def compute_greens_function_matrix(self):
        """
        计算格林函数矩阵
        使用核函数来计算

        返回:
        G: 格林函数矩阵，形状为(3*n_scan_points, 2*n_dipoles)
        """
        if self.dipoles is None or self.scan_points is None:
            raise ValueError("请先设置偶极子平面和扫描平面")

        # 初始化格林函数矩阵
        # 每个扫描点有3个磁场分量，每个偶极子有2个方向（Mx和My）
        G = np.zeros((3 * self.n_scan_points, 2 * self.n_dipoles))

        # 遍历每个扫描点
        for i in range(self.n_scan_points):
            scan_point = self.scan_points[i]

            # 遍历每个偶极子
            for j in range(self.n_dipoles):
                dipole_point = self.dipoles[j]

                # 使用核函数计算格林函数矩阵元素
                K = self.kernel_function(dipole_point[0], scan_point)

                # 从偶极子到扫描点的向量 (mm)
                R = scan_point - dipole_point
                R_norm = np.linalg.norm(R)

                # 如果R_norm太小，可能导致数值不稳定
                if R_norm < 1e-3:  # 小于0.001mm
                    continue

                # 单位向量
                if R_norm > 0:
                    unit_vector = R / R_norm
                else:
                    continue

                # 考虑偶极子位置的缺陷效应
                defect_factor = self.defect_function(dipole_point[0])
                K *= defect_factor

                # Mx产生的场（对应格林函数矩阵的第j*2列）
                # By分量
                G[3 * i + 1, 2 * j] = K * (-unit_vector[2])  # 对应By

                # Bz分量
                G[3 * i + 2, 2 * j] = K * unit_vector[1]  # 对应Bz

                # My产生的场（对应格林函数矩阵的第j*2+1列）
                # Bx分量
                G[3 * i, 2 * j + 1] = K * unit_vector[2]  # 对应Bx

                # Bz分量
                G[3 * i + 2, 2 * j + 1] = K * (-unit_vector[0])  # 对应Bz

        print(f"已计算格林函数矩阵，形状为{G.shape}")
        return G

    def load_measured_data(self, measured_fields):
        """
        加载测量磁场数据

        参数:
        measured_fields: 测量磁场数据，形状为(n_scan_points, 3)
        """
        if measured_fields.shape[0] != self.n_scan_points:
            raise ValueError(f"输入数据点数与扫描点数不符，应为({self.n_scan_points}, 3)")

        # 将数据展平为向量形式
        self.measured_fields = measured_fields.reshape(-1)

        print(f"已加载测量数据，形状为{self.measured_fields.shape}")
        return self.measured_fields

    def predict_magnetic_moments(self):
        """
        预测偶极子的磁矩

        返回:
        magnetic_moments: 预测的磁矩，形状为(n_dipoles, 2)，对应Mx和My
        """
        if self.model is None:
            raise ValueError("请先训练模型")

        # 计算格林函数矩阵
        G = self.compute_greens_function_matrix()

        # 使用模型预测测量值
        X = self.scaler_X.transform(G)
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()

        # 使用伪逆求解线性系统
        # 这相当于求解最小二乘问题: G * m = y_pred
        G_pinv = np.linalg.pinv(G)
        m = G_pinv @ y_pred

        # 重塑为正确的形状
        magnetic_moments = m.reshape(self.n_dipoles, 2)

        print(f"已预测偶极子磁矩，形状为{magnetic_moments.shape}")
        return magnetic_moments

    def predict_magnetic_field(self, test_points):
        """
        预测给定点的磁场分布

        参数:
        test_points: 需要预测磁场的点的坐标，形状为(n_points, 3)，单位为mm

        返回:
        magnetic_field: 预测的磁场，形状为(n_points, 3)，单位为nT
        """
        if self.model is None and not hasattr(self, 'magnetic_moments'):
            raise ValueError("请先训练模型或预测磁矩")

        # 如果还没有预测过磁矩，先预测
        if not hasattr(self, 'magnetic_moments'):
            self.magnetic_moments = self.predict_magnetic_moments()

        # 初始化磁场数组
        n_points = len(test_points)
        magnetic_field = np.zeros((n_points, 3))

        # 计算每个测试点的磁场
        for i, point in enumerate(test_points):
            # 初始化该点的三个磁场分量
            Bx, By, Bz = 0, 0, 0

            # 累加所有偶极子的贡献
            for j in range(self.n_dipoles):
                dipole_pos = self.dipoles[j]
                Mx = self.magnetic_moments[j, 0]
                My = self.magnetic_moments[j, 1]

                # 使用核函数计算磁场贡献
                K = self.kernel_function(dipole_pos[0], point)

                # 从偶极子到测试点的向量
                R = point - dipole_pos
                R_norm = np.linalg.norm(R)

                # 如果距离太小，跳过
                if R_norm < 1e-3:
                    continue

                # 单位向量
                unit_vector = R / R_norm

                # 累加Mx的贡献
                # By分量
                By += Mx * K * (-unit_vector[2])
                # Bz分量
                Bz += Mx * K * unit_vector[1]

                # 累加My的贡献
                # Bx分量
                Bx += My * K * unit_vector[2]
                # Bz分量
                Bz += My * K * (-unit_vector[0])

            # 存储计算结果
            magnetic_field[i] = [Bx, By, Bz]

        return magnetic_field