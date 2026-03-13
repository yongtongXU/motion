import os
import shutil
import subprocess
from datetime import datetime
import csv
import numpy as np


# =========================
# 场景配置：保持与当前 VFH 场景一致
# =========================
WORLD_W, WORLD_H = 1000.0, 1000.0
START = np.array([30.0, 30.0], dtype=float)
GOAL = np.array([950.0, 950.0], dtype=float)
GOAL_TOL = 15.0

# USV 初始姿态/运动约束
INIT_HEADING_DEG = 45.0
INIT_SPEED = 0.0
USV_MAX_SPEED = 8.9
USV_MAX_ACCEL = 2.5
USV_MAX_DECEL = 2.5
USV_MAX_YAW_RATE = 1.2
USV_MAX_YAW_ACCEL = 2.0

USV_RADIUS = 12.0
SAFE_MARGIN = 15.0

OBSTACLES_INIT = [
    {"name": "ship_north", "pos": np.array([500.0, 920.0], dtype=float), "vel": np.array([-0.1, -5.8], dtype=float), "r": 30.0},
    {"name": "ship_south", "pos": np.array([580.0, 80.0], dtype=float), "vel": np.array([0.1, 6.2], dtype=float), "r": 30.0},
    {"name": "ship_east",  "pos": np.array([940.0, 270.0], dtype=float), "vel": np.array([-4.0, -0.1], dtype=float), "r": 30.0},
    {"name": "ship_west",  "pos": np.array([80.0, 200.0], dtype=float), "vel": np.array([5.8, 0.1], dtype=float), "r": 30.0},
]

DT = 0.05
MAX_STEPS = 12000

VO_TIME_HORIZON = 25.0
VO_SPEED_SAMPLES = 7
VO_YAW_SAMPLES = 31
VO_SPEED_WEIGHT = 0.35
VO_HEADING_WEIGHT = 0.15
VO_CLEARANCE_WEIGHT = 0.08


stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
out_dir = os.path.join(os.path.dirname(__file__), f"vo_output_{stamp}")
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


def update_obstacles(obstacles, dt):
    for obs in obstacles:
        obs["pos"] = obs["pos"] + obs["vel"] * dt


def preferred_velocity(usv_pos):
    goal_vec = GOAL - usv_pos
    dist = np.linalg.norm(goal_vec)
    if dist < 1e-9:
        return np.zeros(2), 0.0
    heading = np.arctan2(goal_vec[1], goal_vec[0])
    pref_speed = min(USV_MAX_SPEED, dist / max(VO_TIME_HORIZON * 0.25, DT))
    return pref_speed * goal_vec / dist, heading


def collision_metrics(usv_pos, cand_vel, obs, horizon):
    rel_pos = usv_pos - obs["pos"]
    rel_vel = cand_vel - obs["vel"]
    radius = obs["r"] + USV_RADIUS + SAFE_MARGIN

    a = float(np.dot(rel_vel, rel_vel))
    b = 2.0 * float(np.dot(rel_pos, rel_vel))
    c = float(np.dot(rel_pos, rel_pos) - radius * radius)

    if c <= 0.0:
        return True, 0.0, -np.sqrt(max(-c, 0.0))

    if a < 1e-12:
        return False, np.inf, np.sqrt(max(c, 0.0))

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False, np.inf, np.sqrt(max(c, 0.0))

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    if t2 < 0.0:
        return False, np.inf, np.sqrt(max(c, 0.0))

    t_enter = max(t1, 0.0)
    if t_enter <= horizon and t2 >= 0.0:
        return True, t_enter, -np.sqrt(max(-min(c, 0.0), 0.0))

    t_star = np.clip(-b / (2.0 * a), 0.0, horizon)
    min_sep = np.linalg.norm(rel_pos + rel_vel * t_star) - radius
    return False, np.inf, min_sep


def score_velocity(cand_vel, pref_vel, pref_heading, current_heading, usv_pos, obstacles):
    heading = current_heading if np.linalg.norm(cand_vel) < 1e-9 else np.arctan2(cand_vel[1], cand_vel[0])
    speed = np.linalg.norm(cand_vel)
    min_clearance = np.inf
    for obs in obstacles:
        _, _, clearance = collision_metrics(usv_pos, cand_vel, obs, VO_TIME_HORIZON)
        min_clearance = min(min_clearance, clearance)

    goal_cost = np.linalg.norm(cand_vel - pref_vel)
    speed_cost = USV_MAX_SPEED - speed
    heading_cost = abs(wrap_angle(heading - pref_heading))
    clearance_bonus = max(min_clearance, -50.0)
    return goal_cost + VO_SPEED_WEIGHT * speed_cost + VO_HEADING_WEIGHT * heading_cost - VO_CLEARANCE_WEIGHT * clearance_bonus


def choose_velocity(usv_pos, usv_vel, speed, heading, yaw_rate, obstacles):
    pref_vel, pref_heading = preferred_velocity(usv_pos)

    min_speed = max(0.0, speed - USV_MAX_DECEL * DT)
    max_speed = min(USV_MAX_SPEED, speed + USV_MAX_ACCEL * DT)

    yaw_rate_min = max(-USV_MAX_YAW_RATE, yaw_rate - USV_MAX_YAW_ACCEL * DT)
    yaw_rate_max = min(USV_MAX_YAW_RATE, yaw_rate + USV_MAX_YAW_ACCEL * DT)

    yaw_rate_samples = np.linspace(yaw_rate_min, yaw_rate_max, VO_YAW_SAMPLES)
    speed_samples = np.linspace(min_speed, max_speed, VO_SPEED_SAMPLES)

    candidates = [np.zeros(2)]
    for yaw_rate_cmd in yaw_rate_samples:
        cand_heading = wrap_angle(heading + yaw_rate_cmd * DT)
        direction = np.array([np.cos(cand_heading), np.sin(cand_heading)], dtype=float)
        for cand_speed in speed_samples:
            candidates.append(cand_speed * direction)

    desired_speed = np.clip(np.linalg.norm(pref_vel), min_speed, max_speed)
    desired_yaw_rate = np.clip(wrap_angle(pref_heading - heading) / DT, yaw_rate_min, yaw_rate_max)
    desired_heading = wrap_angle(heading + desired_yaw_rate * DT)
    desired_dir = np.array([np.cos(desired_heading), np.sin(desired_heading)], dtype=float)
    candidates.append(desired_speed * desired_dir)
    candidates.append(limit_vector_norm(pref_vel, max_speed))
    candidates.append(usv_vel)

    safe_candidates = []
    risky_candidates = []
    for cand_vel in candidates:
        collides = False
        first_ttc = np.inf
        min_clearance = np.inf
        for obs in obstacles:
            will_collide, ttc, clearance = collision_metrics(usv_pos, cand_vel, obs, VO_TIME_HORIZON)
            collides = collides or will_collide
            first_ttc = min(first_ttc, ttc)
            min_clearance = min(min_clearance, clearance)

        score = score_velocity(cand_vel, pref_vel, pref_heading, heading, usv_pos, obstacles)
        item = (score, first_ttc, -min_clearance, cand_vel)
        if collides:
            risky_candidates.append(item)
        else:
            safe_candidates.append(item)

    if safe_candidates:
        safe_candidates.sort(key=lambda x: (x[0], x[2]))
        return safe_candidates[0][3], pref_heading, False

    risky_candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
    return risky_candidates[0][3], pref_heading, True


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
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1100">\n')
    svg.append('<rect width="100%" height="100%" fill="white"/>\n')
    svg.append(f'<rect x="{min(x0, x1):.2f}" y="{min(y0, y1):.2f}" width="{abs(x1 - x0):.2f}" height="{abs(y1 - y0):.2f}" fill="none" stroke="rgb(170,170,170)" stroke-width="2"/>\n')

    svg.append(polyline([trans(p) for p in usv_traj], usv_color, 3))
    for idx, hist in enumerate(obstacles_histories):
        svg.append(polyline([trans(p) for p in hist], colors[idx % len(colors)], 2))

    svg.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="7" fill="rgb(0,100,255)"/>\n')
    svg.append(f'<text x="{sx + 10:.2f}" y="{sy - 10:.2f}" font-size="14" fill="rgb(0,100,255)">Start</text>\n')
    svg.append(f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="7" fill="rgb(0,180,0)"/>\n')
    svg.append(f'<text x="{gx + 10:.2f}" y="{gy - 10:.2f}" font-size="14" fill="rgb(0,150,0)">Goal</text>\n')

    for idx, obs in enumerate(obstacles_now):
        color = colors[idx % len(colors)]
        ox, oy = trans(obs["pos"])
        safe_r = obs["r"] + USV_RADIUS + SAFE_MARGIN
        svg.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="{safe_r:.2f}" fill="none" stroke="rgb(255,180,0)" stroke-width="1.5"/>\n')
        svg.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="{obs["r"]:.2f}" fill="none" stroke="{color}" stroke-width="2"/>\n')
        svg.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="4" fill="{color}"/>\n')

    ux, uy = trans(usv_traj[-1])
    svg.append(f'<circle cx="{ux:.2f}" cy="{uy:.2f}" r="{USV_RADIUS:.2f}" fill="none" stroke="rgb(255,120,120)" stroke-width="1.5"/>\n')
    svg.append(f'<circle cx="{ux:.2f}" cy="{uy:.2f}" r="6" fill="{usv_color}"/>\n')
    svg.append('</svg>\n')

    with open(save_path, "w", encoding="utf-8") as f:
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
    safety_color = np.array([255, 180, 0], dtype=np.uint8)
    usv_safety_color = np.array([255, 120, 120], dtype=np.uint8)

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

    def render_frame(t):
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
            obs_r = int(round(OBSTACLES_INIT[j]["r"]))
            safe_r = int(round(OBSTACLES_INIT[j]["r"] + USV_RADIUS + SAFE_MARGIN))
            _draw_circle(img, ox, oy, safe_r, safety_color, fill=False)
            _draw_circle(img, ox, oy, obs_r, c, fill=False)
            _draw_circle(img, ox, oy, 4, c, fill=True)

        sx, sy = trans(START)
        gx, gy = trans(GOAL)
        _draw_circle(img, sx, sy, 6, np.array([0, 100, 255], dtype=np.uint8), fill=True)
        _draw_circle(img, gx, gy, 6, np.array([0, 180, 0], dtype=np.uint8), fill=True)

        ux, uy = trans(usv_traj[t])
        _draw_circle(img, ux, uy, int(round(USV_RADIUS)), usv_safety_color, fill=False)
        _draw_circle(img, ux, uy, 5, usv_color, fill=True)
        return img

    total = len(usv_traj)
    frame_idx = 0
    for t in range(0, total, stride):
        img = render_frame(t)
        ppm_path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.ppm")
        with open(ppm_path, "wb") as f:
            h, w = img.shape[:2]
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        frame_idx += 1

    if (total - 1) % stride != 0:
        img = render_frame(total - 1)
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
    fallback_steps = 0
    step = -1

    for step in range(MAX_STEPS):
        t = step * DT
        if np.linalg.norm(usv_pos - GOAL) <= GOAL_TOL:
            reached = True
            break

        chosen_vel, pref_heading, used_fallback = choose_velocity(usv_pos, usv_vel, speed, heading, yaw_rate, obstacles)
        if used_fallback:
            fallback_steps += 1

        chosen_speed = float(np.linalg.norm(chosen_vel))
        chosen_heading = heading if chosen_speed < 1e-9 else np.arctan2(chosen_vel[1], chosen_vel[0])
        desired_yaw_rate = np.clip(wrap_angle(chosen_heading - heading) / DT, -USV_MAX_YAW_RATE, USV_MAX_YAW_RATE)
        yaw_acc_cmd = np.clip((desired_yaw_rate - yaw_rate) / DT, -USV_MAX_YAW_ACCEL, USV_MAX_YAW_ACCEL)
        yaw_rate = np.clip(yaw_rate + yaw_acc_cmd * DT, -USV_MAX_YAW_RATE, USV_MAX_YAW_RATE)
        heading = wrap_angle(heading + yaw_rate * DT)

        speed_low = max(0.0, speed - USV_MAX_DECEL * DT)
        speed_high = min(USV_MAX_SPEED, speed + USV_MAX_ACCEL * DT)
        speed = float(np.clip(chosen_speed, speed_low, speed_high))
        usv_vel = speed * np.array([np.cos(heading), np.sin(heading)], dtype=float)
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
            "desired_heading_deg": np.rad2deg(pref_heading),
            "heading_error_deg": np.rad2deg(wrap_angle(pref_heading - heading)),
            "dist_to_goal": np.linalg.norm(usv_pos - GOAL),
            "fallback_vo_step": int(used_fallback),
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
    print(f"VO 回退步数: {fallback_steps}")


if __name__ == "__main__":
    main()
