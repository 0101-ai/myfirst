import numpy as np
from scipy.optimize import minimize
import warnings
import ctypes
warnings.filterwarnings('ignore')

# ===================== 加载C++动态库（D盘项目路径） =====================
LIB_PATH = "/mnt/d/python/pythonProject/libmagnetic_core.so"
lib = ctypes.CDLL(LIB_PATH)

# 定义C++函数的参数/返回值类型
lib.biot_savart_wire.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_double,
    ctypes.c_int
]
lib.biot_savart_wire.restype = ctypes.c_double

lib.predict_field.argtypes = [
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64)
]
lib.predict_field.restype = None

class MagneticFieldReconstructor:
    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7
        self.measured_data = None
        self.optimized_params = None

    def set_measured_data(self, data):
        self.measured_data = np.array(data)
        print(f"已加载 {len(data)} 个测量点")

    # C++加速版：单测点磁场计算
    def biot_savart_wire(self, point, wire_start, wire_end, current, n_segments=500):
        point_m = np.array(point, dtype=np.float64) * 1e-3
        wire_start_m = np.array(wire_start, dtype=np.float64) * 1e-3
        wire_end_m = np.array(wire_end, dtype=np.float64) * 1e-3
        B_T = lib.biot_savart_wire(point_m, wire_start_m, wire_end_m, current, n_segments)
        return B_T * 1e6

    # C++加速版：批量磁场预测
    def predict_field(self, current, points, params):
        wire_x_start, wire_y, wire_z, wire_length, current_factor = params
        wire_start = np.array([wire_x_start, wire_y, wire_z])
        wire_end = np.array([wire_x_start + wire_length, wire_y, wire_z])
        effective_current = current * current_factor

        points_m = np.array(points, dtype=np.float64) * 1e-3
        wire_start_m = wire_start * 1e-3
        wire_end_m = wire_end * 1e-3

        if points_m.ndim == 1:
            points_flat = points_m
            n_points = 1
        else:
            points_flat = points_m.ravel()
            n_points = points_m.shape[0]

        results_T = np.zeros(n_points, dtype=np.float64)
        lib.predict_field(
            effective_current,
            points_flat,
            n_points,
            wire_start_m,
            wire_end_m,
            500,
            results_T
        )
        results_uT = results_T * 1e6
        return results_uT if n_points > 1 else results_uT[0]

    def loss_function(self, params):
        if self.measured_data is None:
            raise ValueError("请先设置测量数据！")
        total_error = 0
        for row in self.measured_data:
            current = row[0]
            point = row[1:4]
            measured_B = row[4]
            predicted_B = self.predict_field(current, [point], params)
            relative_error = (predicted_B - measured_B) / measured_B
            total_error += relative_error ** 2
        return total_error

    def optimize_model(self, initial_guess=None, bounds=None):
        if initial_guess is None:
            initial_guess = [0, 0, 0, 100, 1.0]
        print("正在优化模型参数（C++加速版）...")
        result = minimize(
            self.loss_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )
        self.optimized_params = result.x
        self._print_optimization_result(result)
        return result.x

    def _print_optimization_result(self, result):
        # 修复f-string语法错误：提前拼接字符串
        line = '=' * 50
        print(line)
        print("优化完成！最优参数：")
        print(line)
        print(f"  导线起点X坐标:   {result.x[0]:.2f} mm")
        print(f"  导线Y坐标:       {result.x[1]:.2f} mm")
        print(f"  导线Z坐标:       {result.x[2]:.2f} mm")
        print(f"  导线长度:        {result.x[3]:.1f} mm")
        print(f"  电流校正因子:    {result.x[4]:.4f}")
        print(f"  优化残差:        {result.fun:.6f}")
        print(line)
