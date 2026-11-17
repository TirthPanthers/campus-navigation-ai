# Building Navigation: Shortest Distance Finder
### Foundation of AI (ECSCT24301) — Project-Based Learning
**Adani University — Faculty of Engineering Sciences and Technology (FEST)**  
**Team Members:** Rachit Thakor & Tirth Patel

---

## 🎯 How It Works

The system has a clean, focused setup:
1. **Choose Map Source (2 Inputs)**:
   - **Predefined Map**: College Campus Map or Corporate Office Floor Plan.
   - **Add File of Map**: Upload any custom `.txt` or `.csv` grid map file (`0` = walkable, `1` = wall). A sample file is included in `sample_maps/sample_map.txt`.
2. **Give Two Points**:
   - Start Point $(r_1, c_1)$ or named room.
   - Destination Point $(r_2, c_2)$ or named room.
3. **Find Shortest Distance**:
   - Computes the shortest distance using **A\* Search** (with Manhattan distance) or **BFS**.
   - Displays the exact shortest step count, execution time, and clearly traces the shortest path on the map.

---

## 🚀 Running the Application

```powershell
streamlit run app.py
```
