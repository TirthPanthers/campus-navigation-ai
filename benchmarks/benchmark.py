import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root directory is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.environment import generate_benchmark_grid
from core.search import breadth_first_search, a_star_search

def run_benchmarks(output_dir: str = None) -> pd.DataFrame:
    """
    Executes formal empirical benchmarks as planned in Stage 1 Proposal:
    - Small (10x10), Medium (20x20), Large (50x50)
    - Obstacle densities: 10%, 25%, 40%
    - Metrics: Execution Time (ms), Nodes Expanded, Path Length
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)

    sizes = [10, 20, 50]
    densities = [0.10, 0.25, 0.40]
    num_trials = 5

    records = []

    print("=" * 70)
    print("STARTING EMPIRICAL BENCHMARK SUITE (BFS vs A* SEARCH)")
    print("=" * 70)

    for size in sizes:
        for density in densities:
            for trial in range(num_trials):
                seed = 1000 * size + int(density * 100) + trial
                env = generate_benchmark_grid(size=size, obstacle_density=density, seed=seed)
                start = (0, 0)
                goal = (size - 1, size - 1)

                if not env.is_reachable(start, goal):
                    continue

                # Run BFS
                bfs_res = breadth_first_search(env, start, goal)
                # Run A* (Manhattan)
                astar_res = a_star_search(env, start, goal, heuristic="manhattan")

                records.append({
                    "Grid Size": f"{size}x{size}",
                    "Size": size,
                    "Obstacle Density": f"{int(density*100)}%",
                    "Density_Val": density,
                    "Trial": trial + 1,
                    "BFS Nodes Expanded": bfs_res["nodes_expanded"],
                    "A* Nodes Expanded": astar_res["nodes_expanded"],
                    "BFS Time (ms)": bfs_res["execution_time_ms"],
                    "A* Time (ms)": astar_res["execution_time_ms"],
                    "BFS Path Length": bfs_res["path_cost"],
                    "A* Path Length": astar_res["path_cost"],
                    "Efficiency Gain (%)": ((bfs_res["nodes_expanded"] - astar_res["nodes_expanded"]) / max(1, bfs_res["nodes_expanded"])) * 100.0
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Raw benchmark data successfully saved to: {csv_path}")

    # Generate Report Plots
    generate_plots(df, output_dir)
    return df

def generate_plots(df: pd.DataFrame, output_dir: str):
    """Generates publication-ready academic figures for final project report."""
    summary = df.groupby(["Grid Size", "Obstacle Density"])[
        ["BFS Nodes Expanded", "A* Nodes Expanded", "BFS Time (ms)", "A* Time (ms)", "Efficiency Gain (%)"]
    ].mean().reset_index()

    # Figure 1: Nodes Expanded Comparison (Search Efficiency)
    fig, ax = plt.subplots(figsize=(10, 6))
    grid_order = ["10x10", "20x20", "50x50"]
    summary_by_size = df.groupby("Grid Size")[["BFS Nodes Expanded", "A* Nodes Expanded"]].mean().reindex(grid_order)

    x = range(len(grid_order))
    width = 0.35

    ax.bar([p - width/2 for p in x], summary_by_size["BFS Nodes Expanded"], width, label="BFS (Uninformed Baseline)", color="#e74c3c", alpha=0.9)
    ax.bar([p + width/2 for p in x], summary_by_size["A* Nodes Expanded"], width, label="A* Search (Manhattan)", color="#27ae60", alpha=0.9)

    ax.set_title("Search Efficiency: Total Nodes Expanded by Grid Size", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Size (State Space Dimension)", fontsize=12)
    ax.set_ylabel("Mean Number of Nodes Expanded", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(grid_order)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plot1_path = os.path.join(output_dir, "nodes_expanded_comparison.png")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] Saved Plot 1: {plot1_path}")

    # Figure 2: Execution Time Comparison (ms)
    fig, ax = plt.subplots(figsize=(10, 6))
    time_by_size = df.groupby("Grid Size")[["BFS Time (ms)", "A* Time (ms)"]].mean().reindex(grid_order)

    ax.plot(grid_order, time_by_size["BFS Time (ms)"], marker="o", linewidth=2.5, color="#e74c3c", label="BFS Execution Time")
    ax.plot(grid_order, time_by_size["A* Time (ms)"], marker="s", linewidth=2.5, color="#27ae60", label="A* Execution Time")

    ax.set_title("Computational Speed: Execution Time vs Grid Size", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Size", fontsize=12)
    ax.set_ylabel("Mean Execution Time (milliseconds)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)

    plot2_path = os.path.join(output_dir, "execution_time_comparison.png")
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] Saved Plot 2: {plot2_path}")

    # Figure 3: Node Expansion Reduction Percentage
    fig, ax = plt.subplots(figsize=(8, 5))
    eff_by_size = df.groupby("Grid Size")["Efficiency Gain (%)"].mean().reindex(grid_order)
    bars = ax.bar(grid_order, eff_by_size, color="#2980b9", width=0.45)
    ax.set_title("A* Search Efficiency Gain over BFS (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Node Expansion Reduction (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    plot3_path = os.path.join(output_dir, "search_efficiency_gain.png")
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] Saved Plot 3: {plot3_path}")

if __name__ == "__main__":
    run_benchmarks()
