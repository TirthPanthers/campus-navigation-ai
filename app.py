import time
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.environment import (
    CampusMap,
    get_college_campus_map,
    get_corporate_office_map,
    load_map_from_text
)
from core.search import a_star_search, breadth_first_search, dijkstra_search

# Page config
st.set_page_config(
    page_title="Shortest Path Finder | AI Project",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Building Navigation: Shortest Distance Finder")
st.caption(
    "Foundation of AI (ECSCT24301) | BFS · Dijkstra · A* Search | "
    "Adani University FEST — Rachit Thakor & Tirth Patel"
)
st.markdown("---")

# Helper: draw one grid frame
def draw_frame(env, explored_so_far, path, start_pt, goal_pt, title, algo_color=(0.98, 0.90, 0.60)):
    h, w = env.height, env.width
    fig, ax = plt.subplots(figsize=(6, 6))

    # Base grid
    grid_img = np.ones((h, w, 3))
    for r in range(h):
        for c in range(w):
            if env.grid[r, c] == 1:
                grid_img[r, c] = [0.18, 0.22, 0.27]

    # Explored nodes highlight
    for r, c in explored_so_far:
        if (r, c) != start_pt and (r, c) != goal_pt:
            grid_img[r, c] = list(algo_color)

    # Path overlay in blue
    for r, c in path:
        if (r, c) != start_pt and (r, c) != goal_pt:
            grid_img[r, c] = [0.10, 0.50, 0.95]

    ax.imshow(grid_img, origin="upper")
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#c0c0c0", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(range(0, w, max(1, w // 10)))
    ax.set_yticks(range(0, h, max(1, h // 10)))

    # Start marker
    ax.add_patch(patches.Circle((start_pt[1], start_pt[0]), 0.42,
                                 color="#2ecc71", ec="#27ae60", lw=2, zorder=5))
    ax.text(start_pt[1], start_pt[0], "S", color="white", fontsize=11,
            fontweight="bold", ha="center", va="center", zorder=6)

    # Goal marker
    ax.add_patch(patches.Circle((goal_pt[1], goal_pt[0]), 0.42,
                                 color="#e74c3c", ec="#c0392b", lw=2, zorder=5))
    ax.text(goal_pt[1], goal_pt[0], "G", color="white", fontsize=11,
            fontweight="bold", ha="center", va="center", zorder=6)

    # Room labels for predefined maps
    for name, (pr, pc) in env.pois.items():
        if (pr, pc) not in (start_pt, goal_pt) and env.grid[pr, pc] == 0 and h <= 20:
            ax.plot(pc, pr, marker="o", markersize=3, color="#8c7ae6")
            ax.text(pc, pr - 0.46, name.split()[0], fontsize=6.5,
                    ha="center", color="#2f3542", fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig


# Sidebar: Map source
st.sidebar.header("⚙️ Setup")
st.sidebar.subheader("1. Map Source")
map_source = st.sidebar.radio("Input Type", ["Predefined Map", "Upload Map File (.txt / .csv)"])

env = None
if map_source == "Predefined Map":
    preset = st.sidebar.selectbox(
        "Choose Map",
        ["College Campus Map (20x20)", "Corporate Office Floor (20x20)"]
    )
    env = get_college_campus_map() if "Campus" in preset else get_corporate_office_map()
else:
    uploaded = st.sidebar.file_uploader("Upload grid file (0=path, 1=wall)", type=["txt", "csv"])
    if uploaded:
        try:
            grid_array = load_map_from_text(uploaded.read().decode("utf-8"))
            env = CampusMap(grid_array, name=uploaded.name)
            st.sidebar.success("Loaded: {}x{} map".format(env.height, env.width))
        except Exception as e:
            st.sidebar.error("Parse error: {}".format(e))
    else:
        st.sidebar.info("Upload a file, or use sample_maps/sample_map.txt")

# Sidebar: Points
start_pt = goal_pt = None
if env is not None:
    st.sidebar.subheader("2. Start & Destination")
    pois = env.pois
    use_names = False
    if pois:
        mode = st.sidebar.radio("Input mode", ["Room Names", "(Row, Col) Coordinates"])
        use_names = (mode == "Room Names")

    if use_names:
        start_room = st.sidebar.selectbox("Start Room", list(pois.keys()), index=0)
        goal_room  = st.sidebar.selectbox("Destination", list(pois.keys()), index=min(4, len(pois)-1))
        start_pt = pois[start_room]
        goal_pt  = pois[goal_room]
    else:
        st.sidebar.caption("Rows 0-{}  |  Cols 0-{}".format(env.height-1, env.width-1))
        c1, c2 = st.sidebar.columns(2)
        sr = c1.number_input("Start R", 0, env.height-1, 1)
        sc = c2.number_input("Start C", 0, env.width-1, 1)
        gr = c1.number_input("Goal R", 0, env.height-1, env.height-2)
        gc = c2.number_input("Goal C", 0, env.width-1, env.width-2)
        start_pt = (int(sr), int(sc))
        goal_pt  = (int(gr), int(gc))

    # Sidebar: Algorithm
    st.sidebar.subheader("3. Algorithm")
    algo_choice = st.sidebar.selectbox(
        "Choose Algorithm",
        [
            "A* Search (Informed - Manhattan)",
            "Dijkstra's Algorithm (Cost-Based)",
            "Breadth-First Search (BFS - Blind)",
            "Compare All 3 Algorithms"
        ]
    )

    # Sidebar: Animation
    st.sidebar.subheader("4. Animation")
    animate = st.sidebar.toggle("Animate Step-by-Step", value=False)
    delay = 0.03
    step_size = 1
    if animate:
        speed = st.sidebar.select_slider(
            "Animation Speed",
            options=["Very Slow", "Slow", "Medium", "Fast", "Very Fast"],
            value="Medium"
        )
        speed_map = {
            "Very Slow": 0.15, "Slow": 0.07, "Medium": 0.03,
            "Fast": 0.008, "Very Fast": 0.002
        }
        delay = speed_map[speed]
        step_size = st.sidebar.number_input(
            "Redraw every N nodes", min_value=1, max_value=50, value=1,
            help="Set higher for large maps so animation stays fast."
        )


# Main area
if env is None:
    st.info("Select a predefined map or upload a map file from the sidebar to begin.")
elif start_pt is None or goal_pt is None:
    st.info("Set the start and destination points in the sidebar.")
else:
    valid_start = env.is_valid_state(start_pt)
    valid_goal  = env.is_valid_state(goal_pt)

    if not valid_start:
        st.error("Start {} is a wall or out-of-bounds. Choose a walkable cell (value 0).".format(start_pt))
    elif not valid_goal:
        st.error("Goal {} is a wall or out-of-bounds. Choose a walkable cell (value 0).".format(goal_pt))
    elif start_pt == goal_pt:
        st.warning("Start and destination are the same point!")
    else:
        # Run algorithms
        if "Compare All 3" in algo_choice:
            res_astar    = a_star_search(env, start_pt, goal_pt, heuristic="manhattan")
            res_dijkstra = dijkstra_search(env, start_pt, goal_pt)
            res_bfs      = breadth_first_search(env, start_pt, goal_pt)
            res_primary  = res_astar
        elif "A*" in algo_choice:
            res_primary = a_star_search(env, start_pt, goal_pt, heuristic="manhattan")
        elif "Dijkstra" in algo_choice:
            res_primary = dijkstra_search(env, start_pt, goal_pt)
        else:
            res_primary = breadth_first_search(env, start_pt, goal_pt)

        if not res_primary["found"]:
            st.error("No path exists between these two points!")
        else:
            import pandas as pd

            # Pick highlight colour per algorithm
            if "BFS" in res_primary["algorithm"]:
                acolor = (0.98, 0.78, 0.44)
            elif "Dijkstra" in res_primary["algorithm"]:
                acolor = (0.80, 0.60, 0.95)
            else:
                acolor = (0.55, 0.95, 0.75)

            explored   = res_primary["explored_order"]
            final_path = res_primary["path"]

            # ── Two-column layout: Info LEFT  |  Map RIGHT ─────────────────
            col_info, col_map = st.columns([1, 1.6], gap="medium")

            with col_info:
                # Metric cards (stacked vertically in left column)
                st.metric("Shortest Distance", "{} steps".format(res_primary["path_cost"]))
                st.metric("Nodes Explored",    "{}".format(res_primary["nodes_expanded"]))
                st.metric("Time",              "{:.2f} ms".format(res_primary["execution_time_ms"]))
                st.metric("Algorithm",         res_primary["algorithm"].split("(")[0].strip())

                st.markdown("---")

                # Comparison table (3-algorithm mode)
                if "Compare All 3" in algo_choice:
                    st.markdown("**Algorithm Comparison**")
                    df = pd.DataFrame({
                        "Algorithm": ["BFS", "Dijkstra", "A*"],
                        "Distance":  [
                            "{} steps".format(res_bfs["path_cost"]),
                            "{} steps".format(res_dijkstra["path_cost"]),
                            "{} steps".format(res_astar["path_cost"])
                        ],
                        "Nodes": [
                            res_bfs["nodes_expanded"],
                            res_dijkstra["nodes_expanded"],
                            res_astar["nodes_expanded"]
                        ],
                        "ms": [
                            "{:.3f}".format(res_bfs["execution_time_ms"]),
                            "{:.3f}".format(res_dijkstra["execution_time_ms"]),
                            "{:.3f}".format(res_astar["execution_time_ms"])
                        ]
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.markdown("---")

                # Legend
                st.markdown("**Legend**")
                st.markdown(
                    "- 🟢 **S** = Start\n"
                    "- 🔴 **G** = Goal\n"
                    "- 🔵 **Blue** = Shortest Path\n"
                    "- 🟡 Yellow = BFS explored\n"
                    "- 🟣 Purple = Dijkstra explored\n"
                    "- 🟩 Mint = A* explored\n"
                    "- ⬛ Dark = Wall"
                )
                st.markdown("---")

                # Path coordinates
                with st.expander("Step-by-Step Path Coordinates"):
                    st.write("  -->  ".join(["({},{})".format(r, c) for r, c in final_path]))

            with col_map:
                if animate:
                    # === ANIMATION MODE ===
                    st.caption(
                        "Animating {} nodes, then {} path steps.".format(
                            len(explored), res_primary["path_cost"]
                        )
                    )
                    frame_placeholder = st.empty()
                    status_bar = st.empty()

                    # Phase 1: Exploration
                    shown = []
                    for i, node in enumerate(explored):
                        shown.append(node)
                        if i % step_size == 0 or i == len(explored) - 1:
                            fig = draw_frame(
                                env, shown, [],
                                start_pt, goal_pt,
                                "Exploring... {}/{} | {}".format(
                                    i+1, len(explored), res_primary["algorithm"]),
                                algo_color=acolor
                            )
                            frame_placeholder.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            status_bar.progress(
                                int((i + 1) / len(explored) * 100),
                                text="Node {} / {}".format(i+1, len(explored))
                            )
                            time.sleep(delay)

                    # Phase 2: Path reveal
                    status_bar.empty()
                    path_shown = []
                    for j, node in enumerate(final_path):
                        path_shown.append(node)
                        fig = draw_frame(
                            env, explored, path_shown,
                            start_pt, goal_pt,
                            "Path: step {}/{} | {}".format(
                                j+1, len(final_path)-1, res_primary["algorithm"]),
                            algo_color=acolor
                        )
                        frame_placeholder.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        time.sleep(delay * 2)

                    st.success("Done!  {} steps  |  {} nodes explored.".format(
                        res_primary["path_cost"], len(explored)))

                else:
                    # === STATIC MODE ===
                    fig = draw_frame(
                        env, explored, final_path,
                        start_pt, goal_pt,
                        "{} --> {}   ({} steps)".format(
                            start_pt, goal_pt, res_primary["path_cost"]),
                        algo_color=acolor
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)


