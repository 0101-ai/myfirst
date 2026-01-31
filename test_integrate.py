from magnetic_core import MagneticFieldReconstructor
import numpy as np
import time

# 1. 初始化C++加速版重构器
mfr = MagneticFieldReconstructor()

# 2. 模拟测量数据（你最初的3个点）
measured_data = [
    [3.2, 50, 0, 1, 12.57],   # 电流3.2A，点(50,0,1)mm，实测12.57μT
    [3.2, 50, 0, 2, 6.28],    # 点(50,0,2)mm，实测6.28μT
    [3.2, 50, 0, 3, 4.19],    # 点(50,0,3)mm，实测4.19μT
]
mfr.set_measured_data(measured_data)

# 3. 测试模型优化（无任何参数边界）
print("\n=== 测试C++加速版模型优化 ===")
start_time = time.time()
mfr.optimize_model()
end_time = time.time()
print(f"✅ C++加速后优化耗时：{end_time - start_time:.2f} 秒")

# 4. 测试单个点预测
print("\n=== 测试单个点预测 ===")
point = [50, 0, 1]
pred_B = mfr.predict_field(3.2, [point], mfr.optimized_params)
print(f"预测点(50,0,1)mm的磁场：{pred_B:.2f} μT（理论值12.57）")

# 5. 测试批量100个点预测
print("\n=== 测试批量100个点预测 ===")
batch_points = [[50, 0, i] for i in np.linspace(0.1, 10, 100)]
start_time = time.time()
batch_pred = mfr.predict_field(3.2, batch_points, mfr.optimized_params)
end_time = time.time()
print(f"✅ 批量预测100个点耗时：{end_time - start_time:.2f} 秒")
print(f"✅ 第一个点(50,0,0.1)mm预测值：{batch_pred[0]:.2f} μT（理论值125.66）")
