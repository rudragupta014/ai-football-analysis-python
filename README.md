# ⚽ AI Football Analysis System

An end-to-end computer vision and sports analytics platform for automated football match analysis using YOLOv8, ByteTrack, optical flow, clustering, and AI-driven tactical insights.

This project processes football broadcast footage and generates:

- Real-time player and ball tracking
- Team classification
- Ball possession analytics
- Event detection (passes, interceptions, shots)
- Speed and distance estimation
- Passing network visualizations
- Zone heatmaps
- AI-based action suggestions
- Statistical reports and annotated output videos

---

# 📌 Project Highlights

## Core Capabilities

| Feature | Description |
|---|---|
| Player & Ball Detection | Detects players, referees, and ball using YOLOv8 |
| Multi-Object Tracking | Persistent player IDs using ByteTrack |
| Team Identification | Automatic team assignment using K-Means clustering |
| Ball Possession Analysis | Tracks possession by player and team |
| Event Detection | Detects passes, interceptions, and shots |
| Speed & Distance Estimation | Calculates player movement metrics |
| Camera Motion Compensation | Corrects camera movement using optical flow |
| Tactical Analytics | Passing network and spatial zone analysis |
| AI Suggestions | Suggests passes, dribbles, and shooting opportunities |
| Visual Overlays | Draws trails, IDs, metrics, arrows, and heatmaps |

---

# 🎥 Example Pipeline

```text
Football Video
      ↓
YOLOv8 Detection
      ↓
ByteTrack Tracking
      ↓
Team Assignment
      ↓
Ball Possession & Event Detection
      ↓
Perspective Transformation
      ↓
Speed / Distance Analytics
      ↓
Passing Network + Heatmaps
      ↓
AI Tactical Suggestions
      ↓
Annotated Output Video + CSV Reports
```

---

# 🧠 Technologies Used

## Machine Learning & Computer Vision

- OpenCV
- YOLOv8 (`ultralytics`)
- ByteTrack
- Optical Flow (Lucas–Kanade)
- K-Means Clustering
- Homography / Perspective Transformation

## Data & Analytics

- NumPy
- Pandas
- Matplotlib
- NetworkX

## Development Environment

- Python 3.10+
- Jupyter Notebook

---

# 📂 Project Structure

```text
AI_Football-main/
│
├── ai_module/                     # Tactical AI suggestions
│   ├── action_suggester.py
│   └── events_detector.py
│
├── analysis/                      # Analytics & reporting modules
│   ├── enhanced_stats.py
│   ├── run_all_analysis.py
│   ├── visualize_csv.py
│   └── zone_analysis.py
│
├── camera_movement_estimator/     # Optical flow based stabilization
├── player_ball_assigner/          # Ball possession assignment
├── speed_and_distance_estimator/  # Motion analytics
├── team_assigner/                 # Team classification using clustering
├── trackers/                      # YOLO + ByteTrack tracking
├── view_transformer/              # Perspective transformation
├── utils/                         # Helper utilities
│
├── docs/                          # Additional project documentation
├── stubs/                         # Precomputed tracking data
├── output_videos/                 # Generated videos and outputs
├── training/                      # YOLO training notebooks
│
├── main.py                        # Main execution file
├── pass_network.py                # Passing network generation
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI_Football.git
cd AI_Football
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Main Pipeline

```bash
python main.py
```

This executes:

- Object detection
- Tracking
- Team classification
- Possession analysis
- Event detection
- Tactical analysis
- Video annotation

---

## Run Analytics

```bash
python analysis/run_all_analysis.py
```

Generates:

- Statistical summaries
- CSV reports
- Heatmaps
- Passing analytics
- Team comparisons

---

## Generate Passing Network

```bash
python pass_network.py
```

Creates passing relationship graphs and centrality analysis.

---

# 📊 Outputs

## Generated CSV Files

| File | Description |
|---|---|
| `events.csv` | Detected passes, shots, interceptions |
| `player_statistics.csv` | Per-player analytics |
| `comprehensive_stats_report.json` | Match summary report |

---

## Visual Outputs

| Output | Description |
|---|---|
| Annotated Match Video | Tracking, overlays, AI suggestions |
| Passing Network Graph | Team passing structure |
| Ball Heatmap | Ball control zones |
| Player Density Heatmap | Spatial player movement |
| Statistical Charts | Match analytics visualizations |

---

# 🔍 Module Breakdown

## 🎯 Detection & Tracking

### YOLOv8 Detection

Detects:

- Players
- Referees
- Football

The system uses frame-wise inference and converts detections into tracking-ready objects.

### ByteTrack Tracking

Maintains consistent player identities throughout the match.

Benefits:

- Stable IDs
- Occlusion handling
- Re-identification support
- Reduced tracking loss

---

## 🎽 Team Assignment

The system extracts jersey colors and applies K-Means clustering to classify players into teams.

Features:

- Automatic team grouping
- Multi-frame robustness
- Lighting-aware clustering

---

## 🟢 Ball Possession Tracking

The nearest player to the detected ball is assigned possession.

Outputs include:

- Player possession time
- Team possession percentage
- Possession timeline overlays

---

## 📈 Speed & Distance Estimation

Perspective transformation converts image coordinates into real-world coordinates.

Metrics:

- Distance traveled
- Speed in m/s and km/h
- Motion trajectories

---

## 🧠 AI Tactical Suggestions

The AI module evaluates:

- Passing lanes
- Opponent proximity
- Shot opportunities
- Risk vs reward
- Receiver positioning

Suggestions include:

- Best pass option
- Safe dribble direction
- Potential shooting opportunities

---

## 🔗 Passing Network Analysis

Builds player-to-player passing graphs.

Analytics:

- Pass frequency
- Connectivity
- Centrality metrics
- Team structure visualization

---

## 🗺 Zone Analysis

Spatial analysis modules generate:

- Ball heatmaps
- Player density maps
- Tactical occupation zones

Useful for:

- Tactical review
- Pressing analysis
- Shape analysis

---

# 🧪 Research & Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

- Computer Vision Pipelines
- Multi-Object Tracking
- Sports Analytics
- Tactical AI Systems
- Geometric Transformations
- Data Visualization
- Motion Analysis
- Event Detection Systems
- Graph-Based Analytics

---

# 💼 Why This Project Matters

This repository is a strong portfolio project for roles involving:

- Computer Vision
- AI/ML Engineering
- Sports Analytics
- Data Science
- Video Intelligence
- Deep Learning Applications
- Applied Machine Learning

Recruiters and engineers can quickly evaluate:

- System design ability
- End-to-end ML pipeline implementation
- Modular architecture skills
- Real-world computer vision applications
- Data analytics integration

---

# ⚠️ Current Limitations

- Works best with broadcast-style football footage
- Accuracy depends on video quality and camera angle
- Dense occlusions can reduce tracking accuracy
- Tactical AI suggestions are heuristic-based, not reinforcement learned

---

# 🔮 Future Improvements

Potential enhancements:

- Real-time streaming support
- Transformer-based event detection
- Player re-identification models
- Tactical formation recognition
- xG (Expected Goals) analysis
- Multi-camera support
- Web dashboard deployment
- GPU optimization

---

# 📚 Additional Documentation

See the `docs/` directory for:

- Project explanation
- Action suggestion logic
- Development notes
- Experimental analysis

---

# 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# 👤 Author

Rudra Gupta (AI/ML Engineer Computer Vision • Deep Learning • Sports Analytics).
Built as a practical AI + Computer Vision football analytics system showcasing end-to-end sports intelligence workflows.
If you found this project useful, consider starring the repository.

