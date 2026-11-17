import time
from collections import deque
import heapq
from core.heuristics import manhattan_distance, euclidean_distance

def reconstruct_path(came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
    """Reconstructs the optimal sequence of states from start to goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def breadth_first_search(env, start: tuple[int, int], goal: tuple[int, int]) -> dict:
    """
    Breadth-First Search (BFS) [Uninformed Baseline Search Algorithm]
    - Data Structure: FIFO Queue (collections.deque)
    - Exploration Strategy: Level-by-level radial expansion
    - Guarantees: Complete and optimal for uniform step cost (c=1)
    - Drawback: Explores all directions blindly, high node expansion count.
    """
    t0 = time.perf_counter()

    if not env.is_valid_state(start) or not env.is_valid_state(goal):
        return {
            "algorithm": "Breadth-First Search (BFS)",
            "found": False,
            "path": [],
            "path_cost": 0,
            "nodes_expanded": 0,
            "explored_order": [],
            "execution_time_ms": 0.0,
            "peak_frontier_size": 0,
            "status": "Invalid start or goal state (blocked/out-of-bounds)"
        }

    queue = deque([start])
    visited = {start}
    came_from = {}
    explored_order = []
    peak_frontier = 1

    while queue:
        peak_frontier = max(peak_frontier, len(queue))
        current = queue.popleft()
        explored_order.append(current)

        # Goal test upon expansion
        if current == goal:
            t1 = time.perf_counter()
            path = reconstruct_path(came_from, current)
            return {
                "algorithm": "Breadth-First Search (BFS)",
                "found": True,
                "path": path,
                "path_cost": len(path) - 1,
                "nodes_expanded": len(explored_order),
                "explored_order": explored_order,
                "execution_time_ms": (t1 - t0) * 1000.0,
                "peak_frontier_size": peak_frontier,
                "status": "Path found successfully"
            }

        for neighbor in env.get_successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    t1 = time.perf_counter()
    return {
        "algorithm": "Breadth-First Search (BFS)",
        "found": False,
        "path": [],
        "path_cost": 0,
        "nodes_expanded": len(explored_order),
        "explored_order": explored_order,
        "execution_time_ms": (t1 - t0) * 1000.0,
        "peak_frontier_size": peak_frontier,
        "status": "No path exists to goal"
    }


def a_star_search(env, start: tuple[int, int], goal: tuple[int, int], heuristic="manhattan") -> dict:
    """
    A* Search [Informed Heuristic Search Algorithm]
    - Data Structure: Priority Queue / Min-Heap (heapq)
    - Evaluation Function: f(n) = g(n) + h(n)
        * g(n): exact cumulative cost from start to node n
        * h(n): heuristic estimate from node n to goal
    - Heuristic Options: Manhattan (default, strictly admissible for 4-connected grid)
    - Guarantees: Complete and optimal when h(n) is admissible and consistent.
    """
    t0 = time.perf_counter()

    if not env.is_valid_state(start) or not env.is_valid_state(goal):
        return {
            "algorithm": f"A* Search ({heuristic.title()})",
            "found": False,
            "path": [],
            "path_cost": 0,
            "nodes_expanded": 0,
            "explored_order": [],
            "execution_time_ms": 0.0,
            "peak_frontier_size": 0,
            "status": "Invalid start or goal state (blocked/out-of-bounds)"
        }

    h_func = manhattan_distance if heuristic == "manhattan" else euclidean_distance

    # Min-Heap stores tuples: (f_score, h_score, tie_breaker_counter, state)
    # Using h_score as secondary criterion breaks ties toward states closer to the goal.
    counter = 0
    open_heap = []
    initial_h = h_func(start, goal)
    heapq.heappush(open_heap, (initial_h, initial_h, counter, start))

    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    explored_order = []
    peak_frontier = 1

    while open_heap:
        peak_frontier = max(peak_frontier, len(open_heap))
        f, h, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        closed_set.add(current)
        explored_order.append(current)

        # Goal test
        if current == goal:
            t1 = time.perf_counter()
            path = reconstruct_path(came_from, current)
            return {
                "algorithm": f"A* Search ({heuristic.title()})",
                "found": True,
                "path": path,
                "path_cost": len(path) - 1,
                "nodes_expanded": len(explored_order),
                "explored_order": explored_order,
                "execution_time_ms": (t1 - t0) * 1000.0,
                "peak_frontier_size": peak_frontier,
                "status": "Path found successfully"
            }

        for neighbor in env.get_successors(current):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1  # uniform step cost c=1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h_val = h_func(neighbor, goal)
                f_val = tentative_g + h_val
                counter += 1
                heapq.heappush(open_heap, (f_val, h_val, counter, neighbor))

    t1 = time.perf_counter()
    return {
        "algorithm": f"A* Search ({heuristic.title()})",
        "found": False,
        "path": [],
        "path_cost": 0,
        "nodes_expanded": len(explored_order),
        "explored_order": explored_order,
        "execution_time_ms": (t1 - t0) * 1000.0,
        "peak_frontier_size": peak_frontier,
        "status": "No path exists to goal"
    }


def dijkstra_search(env, start: tuple[int, int], goal: tuple[int, int]) -> dict:
    """
    Dijkstra's Shortest Path Algorithm [Semi-Informed Search]
    - Data Structure: Priority Queue / Min-Heap (heapq)
    - Evaluation Function: f(n) = g(n)   (no heuristic — pure cost-so-far)
    - Explores nodes in order of increasing cumulative cost from start.
    - Guarantees: Complete and optimal for non-negative edge weights.
    - Position: Between BFS (blind) and A* (goal-directed).
    - For uniform cost c=1, Dijkstra expands nodes in same order as BFS
      but its structure allows easy extension to weighted graphs.
    """
    t0 = time.perf_counter()

    if not env.is_valid_state(start) or not env.is_valid_state(goal):
        return {
            "algorithm": "Dijkstra's Algorithm",
            "found": False,
            "path": [],
            "path_cost": 0,
            "nodes_expanded": 0,
            "explored_order": [],
            "execution_time_ms": 0.0,
            "peak_frontier_size": 0,
            "status": "Invalid start or goal state (blocked/out-of-bounds)"
        }

    # Min-Heap stores: (g_cost, tie_breaker, state)
    counter = 0
    open_heap = [(0, counter, start)]
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    explored_order = []
    peak_frontier = 1

    while open_heap:
        peak_frontier = max(peak_frontier, len(open_heap))
        g, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        closed_set.add(current)
        explored_order.append(current)

        # Goal test
        if current == goal:
            t1 = time.perf_counter()
            path = reconstruct_path(came_from, current)
            return {
                "algorithm": "Dijkstra's Algorithm",
                "found": True,
                "path": path,
                "path_cost": len(path) - 1,
                "nodes_expanded": len(explored_order),
                "explored_order": explored_order,
                "execution_time_ms": (t1 - t0) * 1000.0,
                "peak_frontier_size": peak_frontier,
                "status": "Path found successfully"
            }

        for neighbor in env.get_successors(current):
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + 1  # uniform step cost c=1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                counter += 1
                heapq.heappush(open_heap, (tentative_g, counter, neighbor))

    t1 = time.perf_counter()
    return {
        "algorithm": "Dijkstra's Algorithm",
        "found": False,
        "path": [],
        "path_cost": 0,
        "nodes_expanded": len(explored_order),
        "explored_order": explored_order,
        "execution_time_ms": (t1 - t0) * 1000.0,
        "peak_frontier_size": peak_frontier,
        "status": "No path exists to goal"
    }
