import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# 将航向角展开成连续曲线，避免在 -180/180 度附近出现跳变，
# 这样更容易看清航向的细微变化。
heading_cont = np.rad2deg(np.unwrap(np.deg2rad(heading.to_numpy())))
heading_rate_deg_s = np.gradient(heading_cont, t.to_numpy())

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
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

ax1.plot(t, heading_cont, linewidth=1.8, color="tab:blue", label="Unwrapped heading")
ax1.plot(t, heading, linewidth=0.9, color="tab:orange", alpha=0.45, label="Raw heading")
ax1.set_ylabel("Heading (deg)")
ax1.set_title("USV Heading Detail")
ax1.minorticks_on()
ax1.grid(True, which="major", linestyle="--", alpha=0.6)
ax1.grid(True, which="minor", linestyle=":", alpha=0.35)
ax1.legend()

ax2.plot(t, heading_rate_deg_s, linewidth=1.2, color="tab:red")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("dHeading/dt")
ax2.minorticks_on()
ax2.grid(True, which="major", linestyle="--", alpha=0.6)
ax2.grid(True, which="minor", linestyle=":", alpha=0.35)

fig.tight_layout()
plt.savefig(os.path.join(current_dir, "heading_curve.png"), dpi=400)
plt.figure()
plt.plot(t, acc)
plt.xlabel("Time (s)")
plt.ylabel("acc")
plt.title("acc")
plt.grid(True)

plt.savefig(os.path.join(current_dir, "acc.png"), dpi=300)

plt.show()
