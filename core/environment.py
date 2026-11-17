import numpy as np
import random
from collections import deque

class CampusMap:
    def __init__(self, grid: np.ndarray, name: str = "Campus Map", pois: dict[str, tuple[int, int]] = None):
        self.grid = np.array(grid, dtype=int)
        self.height, self.width = self.grid.shape
        self.name = name
        self.pois = pois or {}

    def is_valid_state(self, state: tuple[int, int]) -> bool:
        r, c = state
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] == 0

    def get_successors(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        """
        4-connected orthogonal actions: UP, DOWN, LEFT, RIGHT.
        Each move incurs a uniform step cost of 1.
        """
        r, c = state
        # Orthogonal actions: UP (-1, 0), DOWN (1, 0), LEFT (0, -1), RIGHT (0, 1)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid_neighbors = []
        for dr, dc in actions:
            nxt = (r + dr, c + dc)
            if self.is_valid_state(nxt):
                valid_neighbors.append(nxt)
        return valid_neighbors

    def is_reachable(self, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        """Quick check to confirm if goal is reachable from start."""
        if not self.is_valid_state(start) or not self.is_valid_state(goal):
            return False
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            if cur == goal:
                return True
            for nxt in self.get_successors(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return False


def get_college_campus_map() -> CampusMap:
    """
    20x20 Academic Campus Map (Adani University FEST).
    Structured layout with distinct wings, corridors, central courtyard, and designated rooms.
    """
    grid = np.zeros((20, 20), dtype=int)

    # Perimeter boundary walls
    grid[0, :] = 1; grid[-1, :] = 1
    grid[:, 0] = 1; grid[:, -1] = 1

    # North Wing (Labs & Faculty Rooms)
    grid[5, 1:19] = 1
    grid[5, 4] = 0   # Door to Corridor A
    grid[5, 10] = 0  # Door to Central Plaza
    grid[5, 15] = 0  # Door to Corridor B
    grid[1:5, 7] = 1; grid[3, 7] = 0
    grid[1:5, 13] = 1; grid[2, 13] = 0

    # South Wing (Admin, Library, Canteen)
    grid[14, 1:19] = 1
    grid[14, 5] = 0
    grid[14, 14] = 0
    grid[15:19, 9] = 1; grid[17, 9] = 0

    # Central Courtyard Obstacles (Planters/Fountain/Pillars)
    grid[8:12, 8:12] = 1
    # Courtyard pathways around fountain
    grid[9, 9] = 0; grid[9, 10] = 0; grid[10, 9] = 0; grid[10, 10] = 0
    grid[8, 9] = 0; grid[11, 10] = 0

    pois = {
        "Campus Main Gate": (1, 2),
        "AI & Machine Learning Lab": (2, 4),
        "Faculty Cabins": (2, 10),
        "Robotics & IoT Lab": (2, 16),
        "Central Library": (17, 3),
        "Auditorium & Seminar Hall": (9, 2),
        "Student Cafeteria": (17, 15),
        "Administrative Office": (11, 17),
        "Open Courtyard Garden": (7, 10),
        "Parking Bay Area": (18, 1)
    }

    return CampusMap(grid, name="Adani University FEST Campus", pois=pois)


def get_corporate_office_map() -> CampusMap:
    """
    20x20 Corporate Technology Office floor plan.
    Consists of Reception, Conference Rooms, Workstation Bays, Server Room, Pantry.
    """
    grid = np.zeros((20, 20), dtype=int)

    # Outer perimeter
    grid[0, :] = 1; grid[-1, :] = 1
    grid[:, 0] = 1; grid[:, -1] = 1

    # Executive Cabins & Boardroom (Top row)
    grid[4, 1:19] = 1
    grid[4, 4] = 0; grid[4, 10] = 0; grid[4, 16] = 0
    grid[1:4, 6] = 1; grid[2, 6] = 0
    grid[1:4, 13] = 1; grid[2, 13] = 0

    # Central Workstation Cubicle Clusters
    grid[7:10, 3:7] = 1
    grid[7:10, 12:16] = 1

    # Pantry & Cafeteria (Bottom Left)
    grid[14:19, 8] = 1; grid[16, 8] = 0
    grid[14, 1:8] = 1; grid[14, 4] = 0

    # Server Room (Bottom Right - High Security)
    grid[13:19, 14] = 1; grid[15, 14] = 0
    grid[13, 14:19] = 1; grid[13, 16] = 0

    pois = {
        "Main Reception Desk": (1, 2),
        "Boardroom A": (2, 4),
        "CTO Executive Cabin": (2, 10),
        "HR & Operations": (2, 16),
        "Developer Pod Alpha": (8, 9),
        "Developer Pod Beta": (11, 9),
        "Data Center / Server Room": (16, 17),
        "Pantry & Breakroom": (17, 3),
        "Quiet Focus Booth": (12, 2),
        "Town Hall Space": (11, 17)
    }

    return CampusMap(grid, name="Corporate Technology Office", pois=pois)


def load_map_from_text(text: str) -> np.ndarray:
    """
    Parses map data from text or file contents.
    Supports:
      - CSV (0,1,0,...)
      - Space-delimited numbers (0 1 0 ...)
      - Grid characters ('#' or '1' = wall, '.' or '0' = walkable)
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip() and not line.strip().startswith("//")]
    if not lines:
        raise ValueError("File is empty or contains no valid grid lines.")

    rows = []
    for line in lines:
        # Check if CSV
        if "," in line:
            tokens = [t.strip() for t in line.split(",") if t.strip()]
        elif " " in line or "\t" in line:
            tokens = line.split()
        else:
            # Single string of characters, e.g. "01001" or "..#.."
            tokens = list(line)

        row = []
        for t in tokens:
            if t in ("1", "#", "W", "X", "w", "x"):
                row.append(1)
            else:
                row.append(0)
        rows.append(row)

    # Validate rectangular shape
    row_len = len(rows[0])
    for r in rows:
        if len(r) != row_len:
            raise ValueError(f"Inconsistent row length: expected {row_len} columns, but found {len(r)}.")

    return np.array(rows, dtype=int)


def generate_benchmark_grid(size: int = 20, obstacle_density: float = 0.25, seed: int = None) -> CampusMap:
    """
    Benchmark map generator with controllable obstacle density (e.g. 10%, 25%, 40%)
    for systematic stress testing across small (10x10), medium (20x20), and large (50x50) grids.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    grid = np.zeros((size, size), dtype=int)
    num_obstacles = int(size * size * obstacle_density)

    # Place random obstacles
    obstacle_indices = np.random.choice(size * size, size=num_obstacles, replace=False)
    for idx in obstacle_indices:
        r = idx // size
        c = idx % size
        grid[r, c] = 1

    # Keep corners open for clean testing
    grid[0, 0] = 0
    grid[size - 1, size - 1] = 0

    pois = {
        "Start Point (North-West)": (0, 0),
        "Goal Point (South-East)": (size - 1, size - 1)
    }

    return CampusMap(
        grid,
        name=f"Synthetic Grid {size}x{size} (Density {int(obstacle_density * 100)}%)",
        pois=pois
    )
