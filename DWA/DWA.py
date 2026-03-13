import math
import time
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


# =========================
# Utility
# =========================
def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def hypot2(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


# =========================
# Config
# =========================
@dataclass
class DWAConfig:
    # Robot kinematics constraints
    v_min: float = 0.0
    v_max: float = 8.9         # m/s (1000m范围更合理一点)
    w_min: float = -1.2         # rad/s
    w_max: float = 1.2

    a_v_max: float = 2.5        # m/s^2
    a_w_max: float = 2.0        # rad/s^2

    # Sampling resolution in dynamic window
    v_res: float = 0.4
    w_res: float = 0.08

    # Forward simulation
    dt: float = 0.1
    predict_time: float = 3.0

    # Cost weights
    w_goal: float = 1.0
    w_speed: float = 0.2
    w_clearance: float = 1.4

    # Robot safety (meters)
    robot_radius: float = 6.0
    safety_margin: float = 2.0

    # Clearance shaping
    clearance_floor: float = 0.02
    collision_penalty: float = 1e9


@dataclass
class USVState:
    x: float
    y: float
    yaw: float
    v: float
    w: float


# =========================
# Entities
# =========================
class Obstacle:
    def __init__(self, x: float, y: float, vx: float, vy: float, radius: float):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.radius = float(radius)

    def step(self, dt: float, bounds: Tuple[float, float, float, float]):
        xmin, ymin, xmax, ymax = bounds
        self.x += self.vx * dt
        self.y += self.vy * dt

        # # bounce
        # if self.x < xmin + self.radius:
        #     self.x = xmin + self.radius
        #     self.vx *= -1.0
        # elif self.x > xmax - self.radius:
        #     self.x = xmax - self.radius
        #     self.vx *= -1.0

        # if self.y < ymin + self.radius:
        #     self.y = ymin + self.radius
        #     self.vy *= -1.0
        # elif self.y > ymax - self.radius:
        #     self.y = ymax - self.radius
        #     self.vy *= -1.0


class USV:
    """
    Unicycle model:
      x_dot = v cos(yaw)
      y_dot = v sin(yaw)
      yaw_dot = w
    """
    def __init__(self, init_state: USVState, radius: float):
        self.state = init_state
        self.radius = radius
        self.traj = [(init_state.x, init_state.y)]

    def step(self, v_cmd: float, w_cmd: float, dt: float):
        s = self.state
        s.v = v_cmd
        s.w = w_cmd
        s.yaw = wrap_to_pi(s.yaw + s.w * dt)
        s.x += s.v * math.cos(s.yaw) * dt
        s.y += s.v * math.sin(s.yaw) * dt
        self.traj.append((s.x, s.y))


# =========================
# Logger
# =========================
class CSVLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.f = open(filepath, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "t_wall_s",
            "step",
            "x_m",
            "y_m",
            "yaw_deg",
            "v_mps",
            "w_radps",
            "v_cmd_mps",
            "w_cmd_radps",
            "goal_dist_m",
            "min_clearance_m",
            "collided",
        ])
        self.f.flush()

    def log(self,
            t_wall_s: float,
            step: int,
            state: USVState,
            v_cmd: float,
            w_cmd: float,
            goal_dist: float,
            min_clear: float,
            collided: bool):
        self.w.writerow([
            f"{t_wall_s:.3f}",
            step,
            f"{state.x:.3f}",
            f"{state.y:.3f}",
            f"{math.degrees(state.yaw):.3f}",
            f"{state.v:.3f}",
            f"{state.w:.3f}",
            f"{v_cmd:.3f}",
            f"{w_cmd:.3f}",
            f"{goal_dist:.3f}",
            f"{min_clear:.3f}",
            int(collided),
        ])
        # 你如果担心异常退出导致丢数据，可每步flush；如果更关心性能，可改成每N步flush
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class ObstacleCSVLogger:
    def __init__(self, filepath: str):
        self.f = open(filepath, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "t_wall_s",
            "step",
            "obs_id",
            "x_m",
            "y_m",
            "vx_mps",
            "vy_mps",
            "speed_mps",
            "course_deg",
            "radius_m",
        ])
        self.f.flush()

    def log(self,
            t_wall_s: float,
            step: int,
            obs_id: int,
            ob: Obstacle):
        vx, vy = ob.vx, ob.vy
        speed = math.hypot(vx, vy)
        course = math.degrees(math.atan2(vy, vx)) % 360.0

        self.w.writerow([
            f"{t_wall_s:.3f}",
            step,
            obs_id,
            f"{ob.x:.3f}",
            f"{ob.y:.3f}",
            f"{vx:.3f}",
            f"{vy:.3f}",
            f"{speed:.3f}",
            f"{course:.3f}",
            f"{ob.radius:.3f}",
        ])
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass



# =========================
# DWA Planner
# =========================
class DWAPlanner:
    def __init__(self, cfg: DWAConfig):
        self.cfg = cfg

    def _dynamic_window(self, state: USVState) -> Tuple[float, float, float, float]:
        cfg = self.cfg
        Vs = (cfg.v_min, cfg.v_max, cfg.w_min, cfg.w_max)
        Vd = (
            state.v - cfg.a_v_max * cfg.dt,
            state.v + cfg.a_v_max * cfg.dt,
            state.w - cfg.a_w_max * cfg.dt,
            state.w + cfg.a_w_max * cfg.dt,
        )
        v_min = max(Vs[0], Vd[0])
        v_max = min(Vs[1], Vd[1])
        w_min = max(Vs[2], Vd[2])
        w_max = min(Vs[3], Vd[3])
        return v_min, v_max, w_min, w_max

    def _predict_trajectory(self, state: USVState, v: float, w: float) -> np.ndarray:
        cfg = self.cfg
        x, y, yaw = state.x, state.y, state.yaw
        traj = []
        t = 0.0
        while t <= cfg.predict_time + 1e-9:
            traj.append((x, y, yaw))
            yaw = wrap_to_pi(yaw + w * cfg.dt)
            x += v * math.cos(yaw) * cfg.dt
            y += v * math.sin(yaw) * cfg.dt
            t += cfg.dt
        return np.array(traj, dtype=np.float32)

    def _min_distance_to_obstacles(self, traj: np.ndarray, obstacles: List[Obstacle]) -> float:
        cfg = self.cfg
        rr = cfg.robot_radius + cfg.safety_margin
        min_clear = float("inf")

        for ob in obstacles:
            dx = traj[:, 0] - ob.x
            dy = traj[:, 1] - ob.y
            d = np.sqrt(dx * dx + dy * dy) - (rr + ob.radius)
            cmin = float(np.min(d))
            if cmin < min_clear:
                min_clear = cmin
        return min_clear

    def _goal_cost(self, traj: np.ndarray, goal: Tuple[float, float]) -> float:
        gx, gy = goal
        dx = float(traj[-1, 0] - gx)
        dy = float(traj[-1, 1] - gy)
        return math.hypot(dx, dy)

    def plan(self, state: USVState, goal: Tuple[float, float], obstacles: List[Obstacle]) -> Tuple[float, float, np.ndarray, float]:
        cfg = self.cfg
        v_min, v_max, w_min, w_max = self._dynamic_window(state)

        best_u = (0.0, 0.0)
        best_traj = None
        best_cost = float("inf")
        best_min_clear = -1e9

        def make_samples(lo, hi, res, min_n=5):
            if hi <= lo + 1e-9:
                return np.array([lo], dtype=np.float32)
            n = max(min_n, int(math.ceil((hi - lo) / res)) + 1)
            return np.linspace(lo, hi, n, dtype=np.float32)

        v_samples = make_samples(v_min, v_max, cfg.v_res, min_n=5)
        w_samples = make_samples(w_min, w_max, cfg.w_res, min_n=7)

        for v in v_samples:
            for w in w_samples:
                traj = self._predict_trajectory(state, float(v), float(w))
                min_clear = self._min_distance_to_obstacles(traj, obstacles)

                if min_clear <= 0.0:
                    cost = cfg.collision_penalty
                else:
                    c_goal = self._goal_cost(traj, goal)
                    c_speed = (cfg.v_max - float(v))  # 越快越好
                    c_clear = 1.0 / max(min_clear, cfg.clearance_floor)
                    cost = cfg.w_goal * c_goal + cfg.w_speed * c_speed + cfg.w_clearance * c_clear

                if cost < best_cost:
                    best_cost = cost
                    best_u = (float(v), float(w))
                    best_traj = traj
                    best_min_clear = min_clear

        if best_traj is None:
            best_traj = self._predict_trajectory(state, 0.0, 0.0)
            best_min_clear = self._min_distance_to_obstacles(best_traj, obstacles)

        return best_u[0], best_u[1], best_traj, best_min_clear


# =========================
# Visualization + Simulation
# =========================
class Simulator:
    def __init__(self, world_size_m: float = 1000.0, width_px: int = 900, height_px: int = 900):
        self.world_size_m = float(world_size_m)
        self.width = int(width_px)
        self.height = int(height_px)
        self.scale = min(self.width, self.height) / self.world_size_m  # px per meter

        self.world_xmin = 0.0
        self.world_ymin = 0.0
        self.world_xmax = self.world_size_m
        self.world_ymax = self.world_size_m
        self.bounds = (self.world_xmin, self.world_ymin, self.world_xmax, self.world_ymax)

    def w2p(self, x: float, y: float) -> Tuple[int, int]:
        px = int(round(x * self.scale))
        py = int(round(self.height - y * self.scale))
        return px, py

    def draw(self,
             usv: USV,
             obstacles: List[Obstacle],
             goal: Tuple[float, float],
             best_traj: Optional[np.ndarray],
             fps: float,
             step_idx: int,
             min_clear: float,
             collided: bool) -> np.ndarray:
        # White background
        img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        # light grid every 100m
        grid_m = 100.0
        grid_px = int(round(grid_m * self.scale))
        for gx in range(0, self.width + 1, grid_px):
            cv2.line(img, (gx, 0), (gx, self.height), (235, 235, 235), 1)
        for gy in range(0, self.height + 1, grid_px):
            cv2.line(img, (0, gy), (self.width, gy), (235, 235, 235), 1)

        # goal
        gpx, gpy = self.w2p(goal[0], goal[1])
        cv2.circle(img, (gpx, gpy), max(3, int(10 * self.scale)), (0, 160, 0), 2)
        cv2.putText(img, f"GOAL {goal}", (gpx + 8, gpy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 2, cv2.LINE_AA)

        # obstacles
        for i, ob in enumerate(obstacles):
            opx, opy = self.w2p(ob.x, ob.y)
            rr = max(2, int(round(ob.radius * self.scale)))
            cv2.circle(img, (opx, opy), rr, (0, 0, 255), 2)
            # velocity arrow
            tip = self.w2p(ob.x + ob.vx * 10.0, ob.y + ob.vy * 10.0)  # 10s方向提示
            cv2.arrowedLine(img, (opx, opy), tip, (0, 0, 255), 2, tipLength=0.25)
            cv2.putText(img, f"O{i}", (opx + 6, opy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # predicted trajectory (best)
        if best_traj is not None and len(best_traj) > 1:
            pts = [self.w2p(float(p[0]), float(p[1])) for p in best_traj]
            for k in range(len(pts) - 1):
                cv2.line(img, pts[k], pts[k + 1], (255, 180, 0), 2)

        # executed trajectory
        if len(usv.traj) > 1:
            pts = [self.w2p(p[0], p[1]) for p in usv.traj]
            for k in range(len(pts) - 1):
                cv2.line(img, pts[k], pts[k + 1], (40, 40, 40), 2)

        # robot
        s = usv.state
        rpx, rpy = self.w2p(s.x, s.y)
        rr = max(2, int(round(usv.radius * self.scale)))
        cv2.circle(img, (rpx, rpy), rr, (255, 140, 0), 2)

        # heading
        hx = s.x + usv.radius * math.cos(s.yaw)
        hy = s.y + usv.radius * math.sin(s.yaw)
        hpx, hpy = self.w2p(hx, hy)
        cv2.line(img, (rpx, rpy), (hpx, hpy), (255, 140, 0), 2)

        # HUD
        goal_dist = hypot2(s.x - goal[0], s.y - goal[1])
        cv2.putText(img, f"Step: {step_idx}   FPS~{fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, f"USV: x={s.x:.1f} y={s.y:.1f} yaw={math.degrees(s.yaw):.1f}deg",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, f"v={s.v:.2f} m/s   w={s.w:.2f} rad/s   goal_dist={goal_dist:.1f} m",
                    (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, f"min_clearance={min_clear:.2f} m", (10, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

        if collided:
            cv2.putText(img, "COLLISION!", (10, 132),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

        return img


def check_goal_reached(state: USVState, goal: Tuple[float, float], tol_m: float = 12.0) -> bool:
    return hypot2(state.x - goal[0], state.y - goal[1]) <= tol_m


def main():
    cfg = DWAConfig()

    # 1000m range
    sim = Simulator(world_size_m=1000.0, width_px=2500, height_px=2500)
    goal = (950.0, 950.0)  # 目标点保持你要求的(100,100)

    # Init USV
    usv = USV(
        init_state=USVState(x=30.0, y=30.0, yaw=math.radians(0.0), v=0.0, w=0.0),
        radius=cfg.robot_radius
    )

    # Dynamic obstacles (meters, m/s)
    # 更像“货轮”的四条运动：两条航道(东西向 + 南北向)，对向往来
    obstacles = [
        # 航道1：东西向（y≈200），一东一西
        Obstacle(x=80, y=200, vx=5.8, vy=0.1, radius=16.0),  # Eastbound，几乎水平
        Obstacle(x=940, y=270, vx=-4.2, vy=-0.1, radius=14.0),  # Westbound，反向车道（y略偏）

        # 航道2：南北向（x≈520），一北一南
        Obstacle(x=580, y=80, vx=0.1, vy=6.2, radius=16.0),  # Northbound，几乎竖直
        Obstacle(x=500, y=920, vx=-0.1, vy=-5.8, radius=16.0),  # Southbound，反向车道（x略偏）
    ]

    planner = DWAPlanner(cfg)
    logger = CSVLogger("usv_dwa_log.csv")
    obs_logger = ObstacleCSVLogger("obstacles_log.csv")

    win_name = "USV DWA (1000m x 1000m) - White BG - CSV Logging"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    start_wall = time.time()
    last_wall = start_wall
    fps = 0.0
    step_idx = 0

    try:
        while True:
            t0 = time.time()
            step_idx += 1

            # move obstacles
            for i, ob in enumerate(obstacles):
                ob.step(cfg.dt, sim.bounds)

            # === 障碍物日志（新增） ===
            now_wall = time.time()
            for i, ob in enumerate(obstacles):
                obs_logger.log(
                    t_wall_s=now_wall - start_wall,
                    step=step_idx,
                    obs_id=i,
                    ob=ob
                )

            # plan
            v_cmd, w_cmd, best_traj, best_min_clear = planner.plan(usv.state, goal, obstacles)

            # apply
            usv.step(v_cmd, w_cmd, cfg.dt)

            # keep inside bounds
            usv.state.x = clamp(usv.state.x, sim.world_xmin + usv.radius, sim.world_xmax - usv.radius)
            usv.state.y = clamp(usv.state.y, sim.world_ymin + usv.radius, sim.world_ymax - usv.radius)

            # collision check (instantaneous)
            collided = False
            rr = cfg.robot_radius + cfg.safety_margin
            for ob in obstacles:
                d = hypot2(usv.state.x - ob.x, usv.state.y - ob.y)
                if d <= (rr + ob.radius):
                    collided = True
                    break

            # goal dist
            goal_dist = hypot2(usv.state.x - goal[0], usv.state.y - goal[1])

            # log
            now_wall = time.time()
            logger.log(
                t_wall_s=now_wall - start_wall,
                step=step_idx,
                state=usv.state,
                v_cmd=v_cmd,
                w_cmd=w_cmd,
                goal_dist=goal_dist,
                min_clear=best_min_clear,
                collided=collided,
            )

            # fps estimate
            dt_real = now_wall - last_wall
            last_wall = now_wall
            if dt_real > 1e-6:
                fps = 0.9 * fps + 0.1 * (1.0 / dt_real)

            # draw
            img = sim.draw(usv, obstacles, goal, best_traj, fps, step_idx, best_min_clear, collided)
            if check_goal_reached(usv.state, goal, tol_m=12.0):
                cv2.putText(img, "GOAL REACHED", (10, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 160, 0), 3, cv2.LINE_AA)

            cv2.imshow(win_name, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')) or goal_dist < 15 :
                break
            if key == ord('r') :
                # reset
                usv = USV(
                    init_state=USVState(x=30.0, y=30.0, yaw=math.radians(15.0), v=0.0, w=0.0),
                    radius=cfg.robot_radius
                )
                step_idx = 0
                start_wall = time.time()
                last_wall = start_wall

            # basic pacing to dt
            elapsed = time.time() - t0
            sleep_time = cfg.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        logger.close()
        obs_logger.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
