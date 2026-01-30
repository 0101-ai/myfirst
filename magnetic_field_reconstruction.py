import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MagneticFieldReconstructor:
    """
    磁场三维重构器
    基于毕奥-萨伐尔定律，融合测量数据进行模型参数优化
    """

    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7  # 真空磁导率 T·m/A
        self.measured_data = None
        self.optimized_params = None

    def set_measured_data(self, data):
        """
        设置测量数据

        参数:
            data: 列表，每个元素为 [电流(A), x(mm), y(mm), z(mm), |B|(uT)]
        """
        self.measured_data = np.array(data)
        print(f"已加载 {len(data)} 个测量点")

    def biot_savart_wire(self, point, wire_start, wire_end, current, n_segments=500):
        """
        计算有限长直导线在某点产生的磁场强度

        返回: |B| (uT)
        """
        point = np.array(point, dtype=float)
        wire_start = np.array(wire_start, dtype=float)
        wire_end = np.array(wire_end, dtype=float)

        # 导线分段
        t = np.linspace(0, 1, n_segments + 1)
        wire_points = wire_start + np.outer(t, wire_end - wire_start)
        mid_points = (wire_points[:-1] + wire_points[1:]) / 2
        dl = (wire_end - wire_start) / n_segments

        B = np.array([0.0, 0.0, 0.0])

        for mid in mid_points:
            r_vec = (point - mid) * 1e-3  # mm → m
            r = np.linalg.norm(r_vec)

            if r < 1e-6:
                continue

            dl_m = dl * 1e-3  # mm → m
            cross = np.cross(dl_m, r_vec / r)
            dB = self.mu0 * current / (4 * np.pi * r ** 2) * cross
            B += dB

        B_magnitude = np.linalg.norm(B)
        return B_magnitude * 1e6  # T → uT

    def model_predict(self, current, points, params):
        """
        使用当前参数预测磁场

        params: [wire_x_start, wire_y, wire_z, wire_length, current_factor]
        """
        wire_x_start, wire_y, wire_z, wire_length, current_factor = params

        wire_start = np.array([wire_x_start, wire_y, wire_z])
        wire_end = np.array([wire_x_start + wire_length, wire_y, wire_z])

        effective_current = current * current_factor

        predictions = []
        for point in points:
            B = self.biot_savart_wire(point, wire_start, wire_end, effective_current)
            predictions.append(B)

        return np.array(predictions)

    def loss_function(self, params):
        """
        损失函数：预测值与实测值的相对误差平方和
        """
        total_error = 0

        for row in self.measured_data:
            current = row[0]
            point = row[1:4]
            measured_B = row[4]

            predicted_B = self.model_predict(current, [point], params)[0]

            # 使用相对误差
            relative_error = (predicted_B - measured_B) / measured_B
            total_error += relative_error ** 2

        return total_error

    def optimize_model(self, initial_guess=None):
        """
        优化模型参数
        """
        if initial_guess is None:
            # [wire_x_start, wire_y, wire_z, wire_length, current_factor]
            initial_guess = [0, 0, 0, 100, 1.0]

        bounds = [
            (-50, 50),  # wire_x_start
            (-20, 20),  # wire_y
            (-5, 5),  # wire_z
            (50, 200),  # wire_length
            (0.5, 2.0),  # current_factor
        ]

        print("正在优化模型参数...")
        result = minimize(self.loss_function, initial_guess,
                          method='L-BFGS-B', bounds=bounds)

        self.optimized_params = result.x

        print(f"\n{'=' * 50}")
        print("优化完成！最优参数：")
        print(f"{'=' * 50}")
        print(f"  导线起点X坐标:   {result.x[0]:.2f} mm")
        print(f"  导线Y坐标:       {result.x[1]:.2f} mm")
        print(f"  导线Z坐标:       {result.x[2]:.2f} mm")
        print(f"  导线长度:        {result.x[3]:.1f} mm")
        print(f"  电流校正因子:    {result.x[4]:.4f}")
        print(f"  优化残差:        {result.fun:.6f}")
        print(f"{'=' * 50}")

        return result.x

    def predict_field(self, current, points):
        """
        使用优化后的模型预测任意点的磁场
        """
        if self.optimized_params is None:
            raise ValueError("请先调用 optimize_model() 优化模型！")

        return self.model_predict(current, points, self.optimized_params)

    def predict_field_3d(self, current, x_range, y_range, z_range, nx=20, ny=20, nz=20):
        """
        预测三维空间的磁场分布
        """
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(z_range[0], z_range[1], nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        print(f"正在计算 {len(points)} 个空间点的磁场 (电流={current}A)...")

        B_all = self.predict_field(current, points)
        B_mag = B_all.reshape(X.shape)

        print("计算完成！")

        return X, Y, Z, B_mag


def create_measured_data():
    """
    创建你的实测数据
    格式: [电流(A), x(mm), y(mm), z(mm), |B|(uT)]

    假设测量点在导线正上方 (x=50mm为导线中点, y=0)
    """

    # 你的原始数据
    raw_data = {
        # 电流: [1mm高度实测值, 10mm高度实测值, 20mm高度实测值]
        0.3: [1.98, 1.35, 1.08],
        0.9: [5.92, 4.06, 3.21],
        1.5: [8.08, 5.46, 4.35],
        2.4: [12.88, 8.89, 6.98],
        3.2: [18.13, 12.68, 9.98],
    }

    heights = [1, 10, 20]  # mm
    x_measure = 50  # 假设测量点在导线中点正上方
    y_measure = 0

    measured_data = []

    for current, values in raw_data.items():
        for i, height in enumerate(heights):
            measured_data.append([
                current,  # 电流 (A)
                x_measure,  # x (mm)
                y_measure,  # y (mm)
                height,  # z (mm)
                values[i]  # |B| (uT)
            ])

    return measured_data


def compare_results(reconstructor, measured_data):
    """
    对比预测值与实测值，生成论文表格格式的输出
    """

    print("\n" + "=" * 80)
    print("表5-X 磁场重构结果与实测值对比")
    print("=" * 80)

    # 按高度分组
    heights = [1, 10, 20]
    currents = [0.3, 0.9, 1.5, 2.4, 3.2]

    # 表头
    print(f"\n{'电流设定值':<12}", end='')
    for h in heights:
        print(f"{'测量高度 ' + str(h) + 'mm':<30}", end='')
    print()

    print(f"{'I (A)':<12}", end='')
    for h in heights:
        print(f"{'实测(uT)':<10}{'重建(uT)':<10}{'误差%':<10}", end='')
    print()

    print("-" * 102)

    # 数据行
    results = []

    for current in currents:
        print(f"{current:<12.1f}", end='')

        row_results = {'current': current}

        for h in heights:
            # 找到对应的实测数据
            for row in measured_data:
                if row[0] == current and row[3] == h:
                    measured_B = row[4]
                    point = [row[1], row[2], row[3]]

                    # 模型预测
                    predicted_B = reconstructor.predict_field(current, [point])[0]

                    # 计算相对误差
                    error = abs(predicted_B - measured_B) / measured_B * 100

                    print(f"{measured_B:<10.2f}{predicted_B:<10.2f}{error:<10.2f}", end='')

                    row_results[f'h{h}_meas'] = measured_B
                    row_results[f'h{h}_pred'] = predicted_B
                    row_results[f'h{h}_err'] = error
                    break

        print()
        results.append(row_results)

    print("-" * 102)

    # 计算平均误差
    all_errors = []
    for row in results:
        for h in heights:
            all_errors.append(row[f'h{h}_err'])

    print(f"\n统计信息:")
    print(f"  平均相对误差: {np.mean(all_errors):.2f}%")
    print(f"  最大相对误差: {np.max(all_errors):.2f}%")
    print(f"  最小相对误差: {np.min(all_errors):.2f}%")
    print(f"  误差标准差:   {np.std(all_errors):.2f}%")

    return results


def plot_comparison(reconstructor, measured_data):
    """
    绘制预测值与实测值的对比图
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    heights = [1, 10, 20]
    currents = [0.3, 0.9, 1.5, 2.4, 3.2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # ===== 图1: 磁场随电流变化 =====
    ax1 = axes[0, 0]

    for idx, h in enumerate(heights):
        measured_vals = []
        predicted_vals = []

        for current in currents:
            for row in measured_data:
                if row[0] == current and row[3] == h:
                    measured_vals.append(row[4])
                    point = [row[1], row[2], row[3]]
                    pred = reconstructor.predict_field(current, [point])[0]
                    predicted_vals.append(pred)
                    break

        ax1.plot(currents, measured_vals, markers[idx] + '-', color=colors[idx],
                 markersize=8, linewidth=2, label=f'实测值 (h={h}mm)')
        ax1.plot(currents, predicted_vals, markers[idx] + '--', color=colors[idx],
                 markersize=8, linewidth=2, alpha=0.7, label=f'重建值 (h={h}mm)')

    ax1.set_xlabel('电流 I (A)', fontsize=12)
    ax1.set_ylabel('磁场强度 |B| (μT)', fontsize=12)
    ax1.set_title('(a) 磁场强度随电流变化关系', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ===== 图2: 磁场随高度变化 =====
    ax2 = axes[0, 1]

    current_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(currents)))

    for idx, current in enumerate(currents):
        measured_vals = []
        predicted_vals = []

        for h in heights:
            for row in measured_data:
                if row[0] == current and row[3] == h:
                    measured_vals.append(row[4])
                    point = [row[1], row[2], row[3]]
                    pred = reconstructor.predict_field(current, [point])[0]
                    predicted_vals.append(pred)
                    break

        ax2.plot(heights, measured_vals, 'o-', color=current_colors[idx],
                 markersize=8, linewidth=2, label=f'实测 I={current}A')
        ax2.plot(heights, predicted_vals, 's--', color=current_colors[idx],
                 markersize=6, linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('测量高度 z (mm)', fontsize=12)
    ax2.set_ylabel('磁场强度 |B| (μT)', fontsize=12)
    ax2.set_title('(b) 磁场强度随高度衰减关系', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ===== 图3: 预测值 vs 实测值散点图 =====
    ax3 = axes[1, 0]

    all_measured = []
    all_predicted = []

    for row in measured_data:
        current = row[0]
        point = [row[1], row[2], row[3]]
        measured_B = row[4]
        predicted_B = reconstructor.predict_field(current, [point])[0]

        all_measured.append(measured_B)
        all_predicted.append(predicted_B)

    ax3.scatter(all_measured, all_predicted, c='steelblue', s=80, alpha=0.7, edgecolor='white')

    # 理想线 y=x
    max_val = max(max(all_measured), max(all_predicted)) * 1.1
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想拟合线 (y=x)')

    # 计算 R²
    ss_res = np.sum((np.array(all_predicted) - np.array(all_measured)) ** 2)
    ss_tot = np.sum((np.array(all_measured) - np.mean(all_measured)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    ax3.text(0.05, 0.95, f'$R^2$ = {r_squared:.4f}', transform=ax3.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax3.set_xlabel('实测磁场值 (μT)', fontsize=12)
    ax3.set_ylabel('重建磁场值 (μT)', fontsize=12)
    ax3.set_title('(c) 重建值与实测值相关性', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    ax3.set_aspect('equal')

    # ===== 图4: 各测量点误差分布 =====
    ax4 = axes[1, 1]

    errors = []
    labels = []

    for row in measured_data:
        current = row[0]
        h = row[3]
        point = [row[1], row[2], row[3]]
        measured_B = row[4]
        predicted_B = reconstructor.predict_field(current, [point])[0]

        error = (predicted_B - measured_B) / measured_B * 100
        errors.append(error)
        labels.append(f'{current}A\n{h}mm')

    colors_bar = ['green' if abs(e) < 1 else 'orange' if abs(e) < 2 else 'red' for e in errors]

    bars = ax4.bar(range(len(errors)), errors, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.axhline(y=1, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax4.axhline(y=-1, color='red', linewidth=1, linestyle='--', alpha=0.5)

    ax4.set_xticks(range(len(errors)))
    ax4.set_xticklabels(labels, fontsize=8, rotation=45)
    ax4.set_xlabel('测量点 (电流/高度)', fontsize=12)
    ax4.set_ylabel('相对误差 (%)', fontsize=12)
    ax4.set_title('(d) 各测量点重建误差分布', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('磁场重构结果对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig


def plot_3d_field(reconstructor, current=3.2):
    """
    绘制三维磁场分布图
    """

    # 计算三维磁场分布
    X, Y, Z, B_mag = reconstructor.predict_field_3d(
        current=current,
        x_range=(0, 100),
        y_range=(-30, 30),
        z_range=(1, 25),
        nx=25, ny=15, nz=12
    )

    fig = plt.figure(figsize=(16, 12))

    # ===== 图1: XZ平面切片 (y=0) =====
    ax1 = fig.add_subplot(221)

    y_idx = B_mag.shape[1] // 2
    im1 = ax1.contourf(X[:, y_idx, :], Z[:, y_idx, :], B_mag[:, y_idx, :],
                       levels=30, cmap='hot')
    plt.colorbar(im1, ax=ax1, label='|B| (μT)')

    # 标注导线位置
    ax1.plot([0, 100], [0, 0], 'g-', linewidth=4, label='PCB导线')

    ax1.set_xlabel('X (mm)', fontsize=11)
    ax1.set_ylabel('Z - 高度 (mm)', fontsize=11)
    ax1.set_title(f'(a) XZ平面磁场分布 (y=0, I={current}A)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')

    # ===== 图2: XY平面切片 (不同高度) =====
    ax2 = fig.add_subplot(222)

    z_idx = 2  # 约 z=5mm
    z_val = Z[0, 0, z_idx]
    im2 = ax2.contourf(X[:, :, z_idx], Y[:, :, z_idx], B_mag[:, :, z_idx],
                       levels=30, cmap='hot')
    plt.colorbar(im2, ax=ax2, label='|B| (μT)')

    # 标注导线位置
    ax2.plot([0, 100], [0, 0], 'g-', linewidth=4, label='PCB导线')

    ax2.set_xlabel('X (mm)', fontsize=11)
    ax2.set_ylabel('Y (mm)', fontsize=11)
    ax2.set_title(f'(b) XY平面磁场分布 (z={z_val:.1f}mm, I={current}A)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    # ===== 图3: 3D视图 =====
    ax3 = fig.add_subplot(223, projection='3d')

    # 绘制几个高度的等值面
    for z_idx in [1, 4, 8]:
        z_val = Z[0, 0, z_idx]
        ax3.contourf(X[:, :, z_idx], Y[:, :, z_idx], B_mag[:, :, z_idx],
                     zdir='z', offset=z_val, levels=20, cmap='hot', alpha=0.7)

    # 导线
    ax3.plot([0, 100], [0, 0], [0, 0], 'g-', linewidth=4, label='PCB导线')

    ax3.set_xlabel('X (mm)', fontsize=10)
    ax3.set_ylabel('Y (mm)', fontsize=10)
    ax3.set_zlabel('Z (mm)', fontsize=10)
    ax3.set_title(f'(c) 三维磁场分布 (I={current}A)', fontsize=12, fontweight='bold')
    ax3.view_init(elev=25, azim=-60)

    # ===== 图4: 沿导线方向的磁场分布 =====
    ax4 = fig.add_subplot(224)

    # 在不同高度，沿x方向的磁场变化
    y_idx = B_mag.shape[1] // 2
    x_vals = X[:, y_idx, 0]

    z_indices = [0, 3, 6, 9]
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(z_indices)))

    for idx, z_idx in enumerate(z_indices):
        z_val = Z[0, 0, z_idx]
        B_line = B_mag[:, y_idx, z_idx]
        ax4.plot(x_vals, B_line, '-', color=colors[idx], linewidth=2,
                 label=f'z = {z_val:.1f} mm')

    ax4.set_xlabel('X - 沿导线方向 (mm)', fontsize=11)
    ax4.set_ylabel('磁场强度 |B| (μT)', fontsize=11)
    ax4.set_title(f'(d) 沿导线方向磁场分布 (I={current}A)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('三维磁场分布.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig


def plot_field_decay(reconstructor):
    """
    绘制磁场随高度衰减的理论曲线
    """

    fig, ax = plt.subplots(figsize=(10, 7))

    currents = [0.3, 0.9, 1.5, 2.4, 3.2]
    heights = np.linspace(0.5, 30, 100)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(currents)))

    # 实测数据点
    measured_heights = [1, 10, 20]
    measured_data = create_measured_data()

    for idx, current in enumerate(currents):
        # 理论曲线
        B_theory = []
        for h in heights:
            point = [50, 0, h]  # 导线中点正上方
            B = reconstructor.predict_field(current, [point])[0]
            B_theory.append(B)

        ax.plot(heights, B_theory, '-', color=colors[idx], linewidth=2,
                label=f'I = {current} A (理论)')

        # 实测点
        measured_vals = []
        for row in measured_data:
            if row[0] == current:
                measured_vals.append((row[3], row[4]))

        measured_vals = np.array(measured_vals)
        ax.scatter(measured_vals[:, 0], measured_vals[:, 1],
                   color=colors[idx], s=100, marker='o', edgecolor='black',
                   zorder=5)

    ax.set_xlabel('测量高度 z (mm)', fontsize=12)
    ax.set_ylabel('磁场强度 |B| (μT)', fontsize=12)
    ax.set_title('磁场强度随高度衰减关系', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig('磁场衰减曲线.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig


# ============================================================
#                         主程序
# ============================================================

def main():
    """
    主函数：基于你的实测数据进行磁场重构
    """

    print("=" * 60)
    print("     PCB导线三维空间磁场重构")
    print("=" * 60)

    # ===== 1. 加载实测数据 =====
    print("\n[步骤1] 加载实测数据...")
    measured_data = create_measured_data()

    print("\n原始测量数据：")
    print(f"{'电流(A)':<10}{'X(mm)':<10}{'Y(mm)':<10}{'Z(mm)':<10}{'|B|(μT)':<10}")
    print("-" * 50)
    for row in measured_data:
        print(f"{row[0]:<10.1f}{row[1]:<10.1f}{row[2]:<10.1f}{row[3]:<10.1f}{row[4]:<10.2f}")

    # ===== 2. 创建重构器并优化模型 =====
    print("\n[步骤2] 优化模型参数...")
    reconstructor = MagneticFieldReconstructor()
    reconstructor.set_measured_data(measured_data)
    reconstructor.optimize_model()

    # ===== 3. 对比预测与实测 =====
    print("\n[步骤3] 对比预测值与实测值...")
    results = compare_results(reconstructor, measured_data)

    # ===== 4. 绘制对比图 =====
    print("\n[步骤4] 绘制对比图...")
    plot_comparison(reconstructor, measured_data)

    # ===== 5. 绘制三维磁场分布 =====
    print("\n[步骤5] 绘制三维磁场分布...")
    plot_3d_field(reconstructor, current=3.2)

    # ===== 6. 绘制衰减曲线 =====
    print("\n[步骤6] 绘制磁场衰减曲线...")
    plot_field_decay(reconstructor)

    # ===== 7. 预测任意点磁场 =====
    print("\n[步骤7] 预测任意空间点的磁场...")
    print("\n" + "=" * 60)
    print("预测结果（电流 I = 3.2 A）：")
    print("=" * 60)

    test_points = [
        [50, 0, 0.5],  # 更近的位置
        [50, 0, 5],  # 中间位置
        [50, 0, 15],  # 较远位置
        [50, 0, 30],  # 更远位置
        [25, 0, 10],  # 不同x位置
        [75, 0, 10],  # 不同x位置
        [50, 10, 10],  # 不同y位置
    ]

    print(f"\n{'位置 (mm)':<25}{'预测磁场 (μT)':<15}")
    print("-" * 40)

    for point in test_points:
        B = reconstructor.predict_field(3.2, [point])[0]
        print(f"({point[0]}, {point[1]}, {point[2]}){'':<10}{B:<15.2f}")

    print("\n" + "=" * 60)
    print("磁场重构完成！")
    print("=" * 60)

    return reconstructor


if __name__ == "__main__":
    reconstructor = main()