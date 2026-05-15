# AI Football Analysis - Complete Project Explanation

## Overview

This project is a comprehensive AI-powered football (soccer) match analysis system that automatically tracks players, detects events (passes, shots, interceptions), and generates detailed statistics and visualizations from video footage.

---

## 🎯 Core Components

### 1. **Object Detection & Tracking** (`trackers/tracker.py`)
- **Technology**: YOLOv8 (Ultralytics) for object detection
- **Tracking**: ByteTrack algorithm for multi-object tracking
- **What it does**:
  - Detects players, referees, and the ball in each video frame
  - Assigns unique IDs to each player that persist across frames
  - Tracks bounding boxes and positions for all objects
- **Output**: Tracks dictionary containing position and bounding box data for every frame

### 2. **Team Assignment** (`team_assigner/team_assigner.py`)
- **Method**: Multi-frame HSV color sampling + KMeans clustering
- **What it does**:
  - Samples jersey colors from player torsos across multiple frames
  - Uses KMeans clustering to identify two distinct team colors
  - Assigns each player to Team 1 or Team 2 based on jersey color
  - Computes player motion statistics (speed, distance traveled)
- **Output**: Team assignments and motion stats for each player

### 3. **Camera Movement Estimation** (`camera_movement_estimator/`)
- **Method**: Optical flow (Lucas-Kanade)
- **What it does**:
  - Compensates for camera pan/tilt/zoom movements
  - Adjusts player positions to account for camera motion
  - Ensures accurate spatial measurements
- **Output**: Camera movement vectors per frame

### 4. **Ball-Player Assignment** (`player_ball_assigner/`)
- **What it does**:
  - Determines which player has possession of the ball at each frame
  - Uses proximity-based heuristics
- **Output**: `has_ball` flag for players and team possession array

### 5. **Event Detection** (`ai_module/events_detector.py`)
- **What it detects**:
  - **Passes**: Ball movement between players with speed/distance thresholds
  - **Shots**: High-speed ball movements toward goal areas
  - **Interceptions**: Passes where receiving team differs from sending team
- **Method**:
  - Monitors ball speed and travel distance
  - Tracks possession changes
  - Uses sender/receiver lookup windows to attribute passes
  - Includes trajectory-based fallback for missed detections
- **Output**: List of events with frame numbers, player IDs, teams, and positions

### 6. **Action Suggestions** (`ai_module/action_suggester.py`)
- **What it does**:
  - For each frame where a player has the ball, suggests optimal actions
  - Evaluates candidate passes, dribbles, and shots
  - Uses spatial queries (KD-tree) for efficient proximity checks
- **Factors considered**:
  - Distance to teammates/opponents
  - Receiver "openness" (lack of nearby opponents)
  - Line-of-pass clearance (opponents on passing path)
  - Expected Threat (xT) - position value on the pitch
  - Pass success probability (heuristic model)
- **Output**: Top-3 action suggestions per frame with scores and features

### 7. **Visualization** (`visualization.py`)
- **What it draws**:
  - **Player rings**: Thin colored circles around each player (Team 1: white, Team 2: neon green)
  - **Player IDs**: Prominent labels with colored backgrounds for easy identification
  - **Speed & Distance**: Real-time metrics in meters/km per hour
  - **Motion trails**: Faded paths showing recent player movement
  - **Pass arrows**: Team-colored arrows showing pass direction with "INTERCEPT" labels
  - **Shot markers**: Red circles marking shot locations
  - **Camera overlay**: Movement indicators in top-left
  - **Team legend**: Small box in bottom-right showing team colors
  - **Possession banner**: Team ball control percentages

### 8. **Passing Network Analysis** (`pass_network.py`)
- **What it does**:
  - Builds directed graphs of passing relationships
  - Computes network centrality metrics (degree, betweenness, PageRank, etc.)
  - Generates visualizations of passing networks
  - Produces adjacency matrices and edge lists
- **Output**: Network plots, CSV reports, JSON summaries

### 9. **Enhanced Statistics** (`analysis/enhanced_stats.py`)
- **Generates**:
  - Player performance metrics (passes attempted/completed, completion rate)
  - Team statistics (total passes, success rates, average distances)
  - Zone-based analysis (4 ground zones)
  - Time-based trends (quarters, periods)
- **Output**: CSV files and JSON reports

### 10. **Zone Analysis** (`analysis/zone_analysis.py`)
- **What it does**:
  - Divides the pitch into 4 zones (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
  - Analyzes ball presence in each zone
  - Computes player density by team in each zone
- **Output**: Heatmaps and bar charts showing zone activity

---

## 📊 Data Flow

```
Video Input
    ↓
[Object Detection & Tracking] → Tracks (players, ball positions)
    ↓
[Team Assignment] → Team colors, player teams
    ↓
[Camera Movement Estimation] → Adjusted positions
    ↓
[Ball-Player Assignment] → Possession array
    ↓
[Event Detection] → Passes, shots, interceptions
    ↓
[Action Suggestions] (optional) → Recommended actions
    ↓
[Visualization] → Annotated video
    ↓
[Statistics & Analysis] → CSV reports, charts, heatmaps
```

---

## 🔧 Key Parameters & Tuning

### Detection & Tracking
- `--model`: YOLO model path (default: `yolov8s.pt`)
- `--resize_width`: Video resize width for faster processing

### Event Detection
- `--pass_dist_thresh`: Minimum ball travel for a pass (default: 100px)
- `--pass_speed_thresh`: Minimum ball speed for a pass (default: 150 px/s)
- `--shot_speed_thresh`: Minimum ball speed for a shot (default: 500 px/s)
- `--sender_lookup_window`: Frames to look back for pass sender (default: 6)
- `--receiver_lookup_window`: Frames to look ahead for pass receiver (default: 6)
- `--min_receiver_proximity`: Maximum distance for pass receiver (default: 140px)
- `--min_ball_travel_for_pass`: Minimum ball movement (default: 8px)

### Motion & Visualization
- `--meters_per_pixel`: Conversion factor (auto-calculated if not provided)
- `--trail_length`: Number of frames for motion trails (default: 8)
- `--motion_smooth_window`: Smoothing window for player positions (default: 3)

### Action Suggestions
- `--enable_action_suggestions`: Enable action suggestion module
- `--action_search_radius_m`: Maximum distance to consider teammates (default: 25m)
- `--action_openness_radius_m`: Radius to check for opponents around receiver (default: 8m)
- `--action_min_pass_prob`: Minimum pass probability threshold (default: 0.25)
- `--action_xt_weight`: Weight for expected threat in scoring (default: 0.35)
- `--action_intercept_threshold`: Interception probability threshold (default: 0.6)

---

## 📁 Output Structure

```
outputs/
├── videos/                    # Annotated match videos
│   └── output_video_with_possession.mp4
├── analysis/                  # Statistics and reports
│   ├── player_statistics.csv
│   ├── comprehensive_stats_report.json
│   └── ...
├── visualizations/            # Charts and graphs
│   ├── 01_pass_completion_by_player.png
│   ├── 02_team_comparison.png
│   ├── 03_pass_distance_distribution.png
│   └── 04_passes_over_time.png
├── zone_analysis/             # Zone-based analysis
│   ├── ball_presence_by_zone.png
│   └── player_density_by_zone.png
└── pass_network_outputs/     # Passing network analysis
    ├── pass_network_edges.csv
    ├── pass_network_report.json
    └── plots/
```

---

## 🚀 Usage Examples

### Basic Pipeline
```bash
python main.py --video input_videos/match.mp4 --resize_width 720
```

### With Action Suggestions
```bash
python main.py --video input_videos/match.mp4 --enable_action_suggestions
```

### Generate Statistics
```bash
python analysis/enhanced_stats.py --events_path output_videos/events.csv
```

### Create Visualizations
```bash
python analysis/visualize_csv.py --events_path output_videos/events.csv
```

### Zone Analysis
```bash
python analysis/zone_analysis.py --events_path output_videos/events.csv --tracks_path stubs/track_stubs.pkl
```

### Passing Network
```bash
python pass_network.py --events_path output_videos/events.csv --output_dir pass_network_outputs
```

---

## 🎓 Understanding the Metrics

### Player Statistics
- **Passes Attempted**: Total number of passes initiated by the player
- **Passes Completed**: Successful passes (not intercepted)
- **Completion Rate**: Percentage of successful passes
- **Average Pass Distance**: Mean distance of all passes
- **Passes Received**: Number of passes received from teammates

### Team Statistics
- **Total Passes**: All passes made by the team
- **Success Rate**: Percentage of non-intercepted passes
- **Average Pass Distance**: Mean distance of team passes
- **Unique Players**: Number of different players involved in passes

### Zone Statistics
- **Zone 1 (Top-Left)**: Left side of attacking half
- **Zone 2 (Top-Right)**: Right side of attacking half
- **Zone 3 (Bottom-Left)**: Left side of defensive half
- **Zone 4 (Bottom-Right)**: Right side of defensive half

### Network Centrality Metrics
- **Degree**: Total passes (in + out)
- **In-Degree**: Passes received
- **Out-Degree**: Passes made
- **Betweenness**: Importance as a "bridge" in the network
- **PageRank**: Overall importance in passing network
- **Eigenvector**: Importance based on connections to important players

---

## 🔍 Debugging & Inspection

### Check Events CSV
```bash
python debug_inspect.py --events output_videos/events.csv
```

### Test Video Reading
```bash
python test_read_video.py --video input_videos/match.mp4
```

### Inspect First Frame
- Check `debug_first_annotated_frame.jpg` for visualization quality
- Verify player IDs are visible
- Check team colors are distinct
- Ensure speed/distance are in correct units (m, km/h)

---

## 🛠️ Troubleshooting

### Player IDs Not Visible
- Check that `visualization.py` is using the updated drawing code
- Verify player IDs are being drawn with colored backgrounds

### Low Pass Detection
- Reduce `--pass_speed_thresh` and `--pass_dist_thresh`
- Increase `--sender_lookup_window` and `--receiver_lookup_window`

### High False Positives
- Increase `--min_ball_travel_for_pass`
- Increase `--min_receiver_proximity`
- Adjust `--pass_speed_thresh` upward

### Team Assignment Errors
- Ensure good lighting and clear jersey colors
- Check that team colors are distinct in the video

---

## 📚 Future Enhancements

1. **Multi-Camera Homography**: True world coordinates and metric measurements
2. **Re-ID Module**: Stabilize player IDs across occlusions
3. **Learned Classifiers**: Replace heuristics with ML models
4. **Tactical Analysis**: Heatmaps, passing networks, possession segments
5. **Real-Time Streaming**: Lower-latency detection for live analysis

---

## 📝 Notes

- All distances are in pixels unless `meters_per_pixel` is provided
- Speed is converted to km/h when meters_per_pixel is available
- Player IDs are assigned by the tracker and may change if tracking is lost
- Interceptions are detected when pass receiver is from a different team
- Action suggestions are heuristic-based and can be tuned via parameters

