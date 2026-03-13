import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 获取当前目录
# =========================
current_dir = os.getcwd()
print("Current directory:", current_dir)

# =========================
# 文件名
# =========================
file_name = "usv_dwa_log.csv"

# =========================
# 组成完整路径
# =========================
csv_path = os.path.join(current_dir, file_name)
csv_path = os.path.abspath(csv_path)

print("CSV absolute path:", csv_path)

# =========================
# 检查文件
# =========================
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

# =========================
# 读取数据
# =========================
df = pd.read_csv(csv_path)
"""
t_wall_s,step,x_m,y_m,yaw_deg,v_mps,w_radps,v_cmd_mps,w_cmd_radps,goal_dist_m,min_clearance_m,collided
"""
t = df["t_wall_s"]
speed = df["v_mps"]
heading = df["yaw_deg"]
acc = df["v_cmd_mps"]

# =========================
# 航速曲线
# =========================
plt.figure()
plt.plot(t, speed)
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("USV Speed")
plt.grid(True)

plt.savefig(os.path.join(current_dir, "speed_curve.png"), dpi=300)

# =========================
# 航向曲线
# =========================
plt.figure()
plt.plot(t, heading)
plt.xlabel("Time (s)")
plt.ylabel("Heading (deg)")
plt.title("USV Heading")
plt.grid(True)
plt.savefig(os.path.join(current_dir, "heading_curve.png"), dpi=300)
plt.figure()
plt.plot(t, acc)
plt.xlabel("Time (s)")
plt.ylabel("acc")
plt.title("acc")
plt.grid(True)

plt.savefig(os.path.join(current_dir, "acc.png"), dpi=300)

plt.show()