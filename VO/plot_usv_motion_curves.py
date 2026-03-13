import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_csv(base_dir: str) -> str:
    pattern = os.path.join(base_dir, 'vo_output_*', 'trajectory_all.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f'在 {base_dir} 下没有找到 vo_output_*/trajectory_all.csv'
        )
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def validate_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'CSV 缺少必要列: {missing}')


def make_output_dir(csv_path: str, out_dir: str | None) -> str:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    auto_dir = os.path.join(os.path.dirname(csv_path), 'plots')
    os.makedirs(auto_dir, exist_ok=True)
    return auto_dir


def plot_speed(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['usv_speed'], linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('USV Speed Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'usv_speed_curve.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_heading(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['usv_heading_deg'], linewidth=2, label='Actual Heading')
    if 'desired_heading_deg' in df.columns:
        plt.plot(df['time'], df['desired_heading_deg'], linewidth=1.5, linestyle='--', label='Desired Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (deg)')
    plt.title('USV Heading Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'usv_heading_curve.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_speed_heading(df: pd.DataFrame, out_dir: str):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df['time'], df['usv_speed'], linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (m/s)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(df['time'], df['usv_heading_deg'], linewidth=1.5, linestyle='--')
    ax2.set_ylabel('Heading (deg)')

    plt.title('USV Speed and Heading Curves')
    fig.tight_layout()
    save_path = os.path.join(out_dir, 'usv_speed_heading_curve.png')
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description='绘制 USV 航速/航向曲线图')
    parser.add_argument('csv_path', nargs='?', help='trajectory_all.csv 路径；不填则自动寻找最新结果')
    parser.add_argument('--out-dir', help='图片输出文件夹，默认保存在 CSV 同级目录下的 plots 文件夹')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path if args.csv_path else find_latest_csv(base_dir)
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'找不到 CSV 文件: {csv_path}')

    df = pd.read_csv(csv_path)
    validate_columns(df, ['time', 'usv_speed', 'usv_heading_deg'])
    out_dir = make_output_dir(csv_path, args.out_dir)

    speed_png = plot_speed(df, out_dir)
    heading_png = plot_heading(df, out_dir)
    combo_png = plot_speed_heading(df, out_dir)

    print('✅ 曲线图绘制完成')
    print(f'CSV 文件: {csv_path}')
    print(f'输出目录: {out_dir}')
    print(f'航速曲线: {speed_png}')
    print(f'航向曲线: {heading_png}')
    print(f'组合曲线: {combo_png}')


if __name__ == '__main__':
    main()
