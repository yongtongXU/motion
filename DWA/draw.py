import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



usv_csv = "usv_dwa_log.csv"
obs_csv = "obstacles_log.csv"

if not os.path.exists(usv_csv):
    raise FileNotFoundError(f"Missing {usv_csv}")
if not os.path.exists(obs_csv):
    raise FileNotFoundError(f"Missing {obs_csv}")

df_u = pd.read_csv(usv_csv)
df_o = pd.read_csv(obs_csv)

# ---- required columns ----
u_cols = ["t_wall_s", "x_m", "y_m"]
o_cols = ["t_wall_s", "obs_id", "x_m", "y_m", "vx_mps", "vy_mps"]
for c in u_cols:
    if c not in df_u.columns:
        raise ValueError(f"{usv_csv} missing column: {c}")
for c in o_cols:
    if c not in df_o.columns:
        raise ValueError(f"{obs_csv} missing column: {c}")

# numeric + sort
for c in u_cols:
    df_u[c] = pd.to_numeric(df_u[c], errors="coerce")
for c in o_cols:
    df_o[c] = pd.to_numeric(df_o[c], errors="coerce")

df_u = df_u.dropna(subset=u_cols).sort_values("t_wall_s").reset_index(drop=True)
df_o = df_o.dropna(subset=o_cols).sort_values(["obs_id", "t_wall_s"]).reset_index(drop=True)

# USV trajectory
ux = df_u["x_m"].to_numpy()
uy = df_u["y_m"].to_numpy()

# Downsample to keep the plot clean (tune if needed)
usv_ds = max(1, len(df_u) // 5000)
obs_ds = max(1, len(df_o) // 12000)

# ---- one figure ----
plt.figure(figsize=(9, 9))

# USV
plt.plot(ux[::usv_ds], uy[::usv_ds], linewidth=2.0, label="USV")
plt.scatter([ux[0]], [uy[0]], s=40, marker="o", label="USV start")
plt.scatter([ux[-1]], [uy[-1]], s=55, marker="*", label="USV end")

# Obstacles (each obs_id one dashed line)
for obs_id, g in df_o.groupby("obs_id", sort=True):
    gx = g["x_m"].to_numpy()[::obs_ds]
    gy = g["y_m"].to_numpy()[::obs_ds]
    plt.plot(gx, gy, linestyle="--", linewidth=1.6, label=f"Obs {int(obs_id)}")

    # Optional: sparse velocity arrows (comment out if you want pure lines)
    # sample around 25 arrows per obstacle
    n = len(g)
    step = max(1, n // 25)
    gg = g.iloc[::step, :]
    plt.quiver(
        gg["x_m"].to_numpy(),
        gg["y_m"].to_numpy(),
        gg["vx_mps"].to_numpy(),
        gg["vy_mps"].to_numpy(),
        angles="xy",
        scale_units="xy",
        scale=0.6,      # smaller -> longer arrows
        width=0.0025,
        alpha=0.75
    )

plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectories: USV + Obstacles (single plot)")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()

plt.savefig("traj_only_usv_and_obstacles.png", dpi=300)
plt.show()

print("Saved: traj_only_usv_and_obstacles.png")


