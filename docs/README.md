# AI Football Analysis - Documentation

## Quick Links

- **[Complete Project Explanation](PROJECT_EXPLANATION.md)** - Overview of all components and how they work
- **[Action Suggestions Explanation](ACTION_SUGGESTIONS_EXPLANATION.md)** - Detailed explanation of action suggestion module

## Getting Started

1. **Run the main pipeline:**
   ```bash
   python main.py --video input_videos/match.mp4 --resize_width 720
   ```

2. **Generate all analysis:**
   ```bash
   python analysis/run_all_analysis.py
   ```

3. **View results:**
   - Annotated video: `outputs/videos/`
   - Statistics: `outputs/analysis/`
   - Visualizations: `outputs/visualizations/`
   - Zone analysis: `outputs/zone_analysis/`
   - Passing networks: `outputs/pass_network_outputs/`

## Understanding Player IDs

Player IDs are assigned by the tracking system and displayed prominently in the video:
- **Colored background**: Each player ID has a colored box matching their team
- **Location**: Below each player's ring, above speed/distance stats
- **Format**: "ID: [number]" (e.g., "ID: 5", "ID: 12")

These IDs are used throughout all analysis:
- Events CSV: `from_id`, `to_id` columns
- Statistics: Player performance metrics
- Passing networks: Node labels
- Action suggestions: `owner_id`, `target_id`

## Output Structure

```
outputs/
├── videos/                    # Final annotated videos
├── analysis/                  # Statistics and reports
│   ├── events.csv            # All detected events
│   ├── player_statistics.csv # Per-player metrics
│   └── comprehensive_stats_report.json
├── visualizations/           # Charts and graphs
│   ├── 01_pass_completion_by_player.png
│   ├── 02_team_comparison.png
│   ├── 03_pass_distance_distribution.png
│   └── 04_passes_over_time.png
├── zone_analysis/             # Zone-based analysis
│   ├── ball_presence_by_zone.png
│   └── player_density_by_zone.png
├── pass_network_outputs/      # Network analysis
│   ├── pass_network_edges.csv
│   ├── pass_network_report.json
│   └── plots/
└── debug/                     # Debug artifacts
    └── debug_first_annotated_frame.jpg
```

## Key Features Explained

### 1. Player Tracking
- Uses YOLOv8 for detection
- ByteTrack for ID assignment
- IDs persist across frames (unless tracking is lost)

### 2. Team Assignment
- KMeans clustering on jersey colors
- Multi-frame sampling for robustness
- Team 1: White, Team 2: Neon Green

### 3. Event Detection
- Passes: Ball movement between players
- Shots: High-speed ball movements
- Interceptions: Cross-team passes

### 4. Action Suggestions
- Real-time recommendations for ball carrier
- Evaluates passes, dribbles, shots
- Uses spatial queries and heuristics
- See [ACTION_SUGGESTIONS_EXPLANATION.md](ACTION_SUGGESTIONS_EXPLANATION.md) for details

### 5. Zone Analysis
- Divides pitch into 4 zones
- Analyzes ball presence and player density
- Color-coded visualizations

### 6. Statistics
- Player performance metrics
- Team comparisons
- Time-based trends
- Pass completion rates

## Common Questions

**Q: How are player IDs assigned?**
A: The tracker assigns IDs based on detection order and appearance. IDs may change if tracking is lost.

**Q: Why are some passes not detected?**
A: Adjust thresholds: `--pass_speed_thresh`, `--pass_dist_thresh`, `--min_ball_travel_for_pass`

**Q: How accurate are action suggestions?**
A: They're heuristic-based. Tune parameters for your use case. See action suggestions documentation.

**Q: Can I use my own video?**
A: Yes! Place it in `input_videos/` and update the path in the command.

**Q: How do I identify players in the video?**
A: Player IDs are displayed with colored backgrounds below each player. Match IDs from video to CSV data.

## Troubleshooting

See [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md) for detailed troubleshooting guide.

## Next Steps

1. Review the annotated video and verify player IDs are visible
2. Check `outputs/analysis/events.csv` for detected events
3. Run `analysis/run_all_analysis.py` for comprehensive statistics
4. Explore visualizations in `outputs/visualizations/`
5. Analyze passing networks and zone activity

