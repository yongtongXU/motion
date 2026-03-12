import os
import shutil
import subprocess
from datetime import datetime

import csv
import numpy as np


# =========================
# 场景配置（按需求）
# =========================
WORLD_W, WORLD_H = 1000.0, 1000.0
START = np.array([30.0, 30.0], dtype=float)
GOAL = np.array([950.0, 950.0], dtype=float)
GOAL_TOL = 15.0

# USV 初始姿态/运动约束
INIT_HEADING_DEG = 15.0      # 初始航向角（度），0度指向+x方向，逆时针为正
INIT_SPEED = 0.0             # 初始航速（m/s）
USV_MAX_SPEED = 8.9
USV_MAX_ACCEL = 2.5          # 最大加速（m/s^2）
USV_MAX_DECEL = 2.5          # 最大减速（m/s^2）
USV_MAX_YAW_RATE = np.deg2rad(18.0)    # 最大角速度（rad/s）
USV_MAX_YAW_ACCEL = np.deg2rad(25.0)   # 最大角加速度（rad/s^2）
YAW_P_GAIN = 1.8                         # 航向角控制比例增益

USV_RADIUS = 12.0
SAFE_MARGIN = 20.0

# 4个运动障碍物：海上十字路口（正南/正北/正东/正西各一艘）
# 按“靠右通行”布置航道，并保持直线持续航行（不掉头、无边界反弹）
OBSTACLES_INIT = [
    {"name": "ship_north", "pos": np.array([500.0, 920.0], dtype=float), "vel": np.array([-0.1, -5.8], dtype=float), "r": 30.0},
    {"name": "ship_south", "pos": np.array([580.0, 80.0], dtype=float), "vel": np.array([0.1, 6.2], dtype=float), "r": 30.0},
    {"name": "ship_east",  "pos": np.array([940.0, 270.0], dtype=float), "vel": np.array([-4.0, -0.1], dtype=float), "r": 28.0},
    {"name": "ship_west",  "pos": np.array([80.0, 200.0], dtype=float), "vel": np.array([5.8, 0.1], dtype=float), "r": 28.0},
]

DT = 0.05
MAX_STEPS = 12000

ATTR_GAIN = 2.4
REP_GAIN = 22000.0
REP_INFLUENCE_DIST = 150.0
PREDICT_HORIZON = 1.2


stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
out_dir = os.path.join(os.path.dirname(__file__), f"vfh_output_{stamp}")
os.makedirs(out_dir, exist_ok=True)


def clamp_world(p):
    p[0] = np.clip(p[0], 0.0, WORLD_W)
    p[1] = np.clip(p[1], 0.0, WORLD_H)
    return p


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def limit_vector_norm(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm < 1e-12 or norm <= max_norm:
        return vec
    return vec / norm * max_norm


def limit_scalar_with_decel(value, max_accel, max_decel):
    """标量加速度约束：正向加速受 max_accel，反向减速受 max_decel。"""
    if value >= 0.0:
        return min(value, max_accel)
    return max(value, -max_decel)


def attractive_force(pos, goal):
    d = goal - pos
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        return np.zeros(2)
    return ATTR_GAIN * d / dist


def repulsive_force(usv_pos, usv_vel, obstacles):
    f_rep = np.zeros(2)
    for obs in obstacles:
        obs_pred = obs["pos"] + obs["vel"] * PREDICT_HORIZON
        rel = usv_pos - obs_pred
        dist = np.linalg.norm(rel)
        safe_dist = obs["r"] + USV_RADIUS + SAFE_MARGIN

        if dist < 1e-6:
            continue

        if dist < REP_INFLUENCE_DIST:
            scale = REP_GAIN * (1.0 / dist - 1.0 / REP_INFLUENCE_DIST) / (dist * dist)
            scale = max(scale, 0.0)
            f_rep += scale * (rel / dist)

        if dist < safe_dist * 1.25 and np.linalg.norm(usv_vel) > 1e-6:
            f_rep += -0.6 * usv_vel / np.linalg.norm(usv_vel)

    return f_rep


def update_obstacles(obstacles, dt):
    """船舶式直线持续航行：不掉头、无边界反弹。"""
    for obs in obstacles:
        obs["pos"] = obs["pos"] + obs["vel"] * dt


def draw_all_trajectories_svg(usv_traj, obstacles_histories, obstacles_now, save_path):
    canvas_size = 1100
    pad = 50

    def trans(p):
        x = p[0] + pad
        y = canvas_size - (p[1] + pad)
        return x, y

    def polyline(points, color, width):
        if len(points) < 2:
            return ""
        s = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
        return f'<polyline points="{s}" fill="none" stroke="{color}" stroke-width="{width}" />\n'

    usv_color = "rgb(220,30,30)"
    colors = ["rgb(255,140,0)", "rgb(0,170,0)", "rgb(160,0,160)", "rgb(0,120,220)"]

    sx, sy = trans(START)
    gx, gy = trans(GOAL)
    x0, y0 = trans(np.array([0, 0]))
    x1, y1 = trans(np.array([WORLD_W, WORLD_H]))

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}">\n')
    svg.append('<rect width="100%" height="100%" fill="white"/>\n')
    svg.append(f'<rect x="{min(x0,x1):.2f}" y="{min(y0,y1):.2f}" width="{abs(x1-x0):.2f}" height="{abs(y1-y0):.2f}" fill="none" stroke="rgb(170,170,170)" stroke-width="2"/>\n')

    usv_points = [trans(p) for p in usv_traj]
    svg.append(polyline(usv_points, usv_color, 3))

    for idx, hist in enumerate(obstacles_histories):
        obs_points = [trans(p) for p in hist]
        svg.append(polyline(obs_points, colors[idx % len(colors)], 2))

    svg.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="7" fill="rgb(0,100,255)"/>\n')
    svg.append(f'<text x="{sx+10:.2f}" y="{sy-10:.2f}" font-size="14" fill="rgb(0,100,255)">Start</text>\n')
    svg.append(f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="7" fill="rgb(0,180,0)"/>\n')
    svg.append(f'<text x="{gx+10:.2f}" y="{gy-10:.2f}" font-size="14" fill="rgb(0,150,0)">Goal</text>\n')

    for idx, obs in enumerate(obstacles_now):
        color = colors[idx % len(colors)]
        ox, oy = trans(obs["pos"])
        r = obs["r"]
        svg.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="{r:.2f}" fill="none" stroke="{color}" stroke-width="2"/>\n')
        svg.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="4" fill="{color}"/>\n')
        svg.append(f'<text x="{ox+8:.2f}" y="{oy-8:.2f}" font-size="13" fill="{color}">{obs["name"]}</text>\n')

    ux, uy = trans(usv_traj[-1])
    svg.append(f'<circle cx="{ux:.2f}" cy="{uy:.2f}" r="6" fill="{usv_color}"/>\n')

    svg.append(f'<text x="20" y="25" font-size="14" fill="{usv_color}">USV Trajectory</text>\n')
    for idx in range(len(obstacles_histories)):
        color = colors[idx % len(colors)]
        svg.append(f'<text x="20" y="{50 + idx*20}" font-size="14" fill="{color}">Obstacle {idx+1} Trajectory</text>\n')

    svg.append('</svg>\n')

    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(svg)


def _draw_line(img, x0, y0, x1, y1, color, thickness=1):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n <= 1:
        if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
            img[y0, x0] = color
        return
    xs = np.linspace(x0, x1, n).astype(int)
    ys = np.linspace(y0, y1, n).astype(int)
    h, w = img.shape[:2]
    for x, y in zip(xs, ys):
        x_min = max(0, x - thickness)
        x_max = min(w, x + thickness + 1)
        y_min = max(0, y - thickness)
        y_max = min(h, y + thickness + 1)
        img[y_min:y_max, x_min:x_max] = color


def _draw_circle(img, cx, cy, r, color, fill=False):
    h, w = img.shape[:2]
    x_min = max(0, cx - r)
    x_max = min(w - 1, cx + r)
    y_min = max(0, cy - r)
    y_max = min(h - 1, cy + r)
    yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if fill:
        mask = dist2 <= r * r
    else:
        mask = (dist2 <= r * r) & (dist2 >= (r - 1) * (r - 1))
    img[y_min:y_max + 1, x_min:x_max + 1][mask] = color


def export_animation_gif(usv_traj, obstacles_histories, out_dir, fps=20, stride=4):
    canvas_size = 1100
    pad = 50

    def trans(p):
        x = int(p[0] + pad)
        y = int(canvas_size - (p[1] + pad))
        return x, y

    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    usv_color = np.array([220, 30, 30], dtype=np.uint8)
    colors = [
        np.array([255, 140, 0], dtype=np.uint8),
        np.array([0, 170, 0], dtype=np.uint8),
        np.array([160, 0, 160], dtype=np.uint8),
        np.array([0, 120, 220], dtype=np.uint8),
    ]

    total = len(usv_traj)
    frame_idx = 0
    for t in range(0, total, stride):
        img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        x0, y0 = trans(np.array([0, 0]))
        x1, y1 = trans(np.array([WORLD_W, WORLD_H]))
        _draw_line(img, x0, y0, x1, y0, np.array([170, 170, 170], dtype=np.uint8))
        _draw_line(img, x1, y0, x1, y1, np.array([170, 170, 170], dtype=np.uint8))
        _draw_line(img, x1, y1, x0, y1, np.array([170, 170, 170], dtype=np.uint8))
        _draw_line(img, x0, y1, x0, y0, np.array([170, 170, 170], dtype=np.uint8))

        for i in range(1, t + 1):
            x_prev, y_prev = trans(usv_traj[i - 1])
            x_cur, y_cur = trans(usv_traj[i])
            _draw_line(img, x_prev, y_prev, x_cur, y_cur, usv_color, thickness=1)

        for j, hist in enumerate(obstacles_histories):
            c = colors[j % len(colors)]
            end_t = min(t, len(hist) - 1)
            for i in range(1, end_t + 1):
                x_prev, y_prev = trans(hist[i - 1])
                x_cur, y_cur = trans(hist[i])
                _draw_line(img, x_prev, y_prev, x_cur, y_cur, c, thickness=1)

            ox, oy = trans(hist[end_t])
            _draw_circle(img, ox, oy, 4, c, fill=True)

        sx, sy = trans(START)
        gx, gy = trans(GOAL)
        _draw_circle(img, sx, sy, 6, np.array([0, 100, 255], dtype=np.uint8), fill=True)
        _draw_circle(img, gx, gy, 6, np.array([0, 180, 0], dtype=np.uint8), fill=True)

        ux, uy = trans(usv_traj[t])
        _draw_circle(img, ux, uy, 5, usv_color, fill=True)

        ppm_path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.ppm")
        with open(ppm_path, "wb") as f:
            h, w = img.shape[:2]
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        frame_idx += 1

    if (total - 1) % stride != 0:
        t = total - 1
        img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        for i in range(1, t + 1):
            x_prev, y_prev = trans(usv_traj[i - 1])
            x_cur, y_cur = trans(usv_traj[i])
            _draw_line(img, x_prev, y_prev, x_cur, y_cur, usv_color, thickness=1)
        for j, hist in enumerate(obstacles_histories):
            c = colors[j % len(colors)]
            for i in range(1, len(hist)):
                x_prev, y_prev = trans(hist[i - 1])
                x_cur, y_cur = trans(hist[i])
                _draw_line(img, x_prev, y_prev, x_cur, y_cur, c, thickness=1)
        ppm_path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.ppm")
        with open(ppm_path, "wb") as f:
            h, w = img.shape[:2]
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(img.tobytes())

    gif_path = os.path.join(out_dir, "all_trajectories.gif")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, "未找到 ffmpeg，已保留 PPM 序列帧。"

    cmd = [
        ffmpeg,
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%05d.ppm"),
        "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        gif_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return gif_path, None
    except subprocess.CalledProcessError:
        return None, "ffmpeg 生成 GIF 失败，已保留 PPM 序列帧。"


def main():
    usv_pos = START.copy()
    heading = np.deg2rad(INIT_HEADING_DEG)
    yaw_rate = 0.0
    speed = float(np.clip(INIT_SPEED, 0.0, USV_MAX_SPEED))
    usv_vel = speed * np.array([np.cos(heading), np.sin(heading)], dtype=float)

    obstacles = [
        {
            "name": d["name"],
            "pos": d["pos"].copy(),
            "vel": d["vel"].copy(),
            "r": float(d["r"]),
        }
        for d in OBSTACLES_INIT
    ]

    usv_traj = [usv_pos.copy()]
    obstacles_histories = [[obs["pos"].copy()] for obs in obstacles]

    records = []
    reached = False

    for step in range(MAX_STEPS):
        t = step * DT
        if np.linalg.norm(usv_pos - GOAL) <= GOAL_TOL:
            reached = True
            break

        f_attr = attractive_force(usv_pos, GOAL)
        f_rep = repulsive_force(usv_pos, usv_vel, obstacles)
        f_total = f_attr + f_rep

        e_forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)

        # 1) 纵向速度变化：同时考虑加速与减速约束
        acc_long_des = float(np.dot(f_total, e_forward))
        acc_long_cmd = limit_scalar_with_decel(acc_long_des, USV_MAX_ACCEL, USV_MAX_DECEL)
        speed = np.clip(speed + acc_long_cmd * DT, 0.0, USV_MAX_SPEED)

        # 2) 航向变化：由 APF 合力方向生成期望航向，再施加角速度/角加速度限制
        if np.linalg.norm(f_total) > 1e-9:
            desired_heading = np.arctan2(f_total[1], f_total[0])
        elif speed > 1e-9:
            desired_heading = heading
        else:
            goal_vec = GOAL - usv_pos
            desired_heading = np.arctan2(goal_vec[1], goal_vec[0]) if np.linalg.norm(goal_vec) > 1e-9 else heading

        heading_error = wrap_angle(desired_heading - heading)
        desired_yaw_rate = np.clip(YAW_P_GAIN * heading_error, -USV_MAX_YAW_RATE, USV_MAX_YAW_RATE)
        yaw_acc_cmd = np.clip((desired_yaw_rate - yaw_rate) / DT, -USV_MAX_YAW_ACCEL, USV_MAX_YAW_ACCEL)
        yaw_rate = np.clip(yaw_rate + yaw_acc_cmd * DT, -USV_MAX_YAW_RATE, USV_MAX_YAW_RATE)
        heading = wrap_angle(heading + yaw_rate * DT)

        # 3) 根据新的航向和航速更新速度与位置
        e_forward = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        usv_vel = speed * e_forward
        usv_pos = clamp_world(usv_pos + usv_vel * DT)
        update_obstacles(obstacles, DT)

        usv_traj.append(usv_pos.copy())
        for i, obs in enumerate(obstacles):
            obstacles_histories[i].append(obs["pos"].copy())

        row = {
            "time": t,
            "usv_x": usv_pos[0],
            "usv_y": usv_pos[1],
            "usv_vx": usv_vel[0],
            "usv_vy": usv_vel[1],
            "usv_speed": speed,
            "usv_heading_rad": heading,
            "usv_heading_deg": np.rad2deg(heading),
            "usv_yaw_rate_rad_s": yaw_rate,
            "usv_yaw_rate_deg_s": np.rad2deg(yaw_rate),
            "usv_yaw_acc_rad_s2": yaw_acc_cmd,
            "usv_yaw_acc_deg_s2": np.rad2deg(yaw_acc_cmd),
            "usv_ax": acc_long_cmd * e_forward[0],
            "usv_ay": acc_long_cmd * e_forward[1],
            "usv_acc": abs(acc_long_cmd),
            "usv_acc_longitudinal": acc_long_cmd,
            "desired_heading_deg": np.rad2deg(desired_heading),
            "heading_error_deg": np.rad2deg(heading_error),
            "dist_to_goal": np.linalg.norm(usv_pos - GOAL),
        }
        for i, obs in enumerate(obstacles, start=1):
            row[f"obs{i}_x"] = obs["pos"][0]
            row[f"obs{i}_y"] = obs["pos"][1]
            row[f"obs{i}_vx"] = obs["vel"][0]
            row[f"obs{i}_vy"] = obs["vel"][1]
            row[f"obs{i}_r"] = obs["r"]
        records.append(row)

    csv_path = os.path.join(out_dir, "trajectory_all.csv")
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("time,usv_x,usv_y\n")

    img_path = os.path.join(out_dir, "all_trajectories.svg")
    draw_all_trajectories_svg(usv_traj, obstacles_histories, obstacles, img_path)

    gif_path, gif_err = export_animation_gif(usv_traj, obstacles_histories, out_dir, fps=20, stride=4)

    print("✅ 仿真完成")
    print(f"输出目录: {out_dir}")
    print(f"轨迹文件: {csv_path}")
    print(f"结果图片: {img_path}")
    if gif_path:
        print(f"动图文件: {gif_path}")
    else:
        print(f"动图生成提示: {gif_err}")
    print(f"是否到达目标: {reached}")
    print(f"总步数: {step + 1}")


if __name__ == "__main__":
    main()
