"""
Ground Zone Analysis

Divides the football pitch into 4 zones and analyzes:
- Ball presence in each zone
- Player density by team in each zone
- Pass origins and destinations by zone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from typing import Dict, List, Tuple
import cv2

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[WARN] seaborn not available, using default matplotlib style")

# Configure matplotlib for better fonts and clarity
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Zone colors for visualization
ZONE_COLORS = {
    1: (255, 100, 100),    # Top-Left: Red
    2: (100, 255, 100),    # Top-Right: Green
    3: (100, 100, 255),    # Bottom-Left: Blue
    4: (255, 255, 100),    # Bottom-Right: Yellow
}

ZONE_NAMES = {
    1: "Top-Left",
    2: "Top-Right",
    3: "Bottom-Left",
    4: "Bottom-Right",
}


def get_zone(x: float, y: float, frame_width: int, frame_height: int) -> int:
    """Determine which zone a point belongs to (1-4)."""
    mid_x = frame_width / 2
    mid_y = frame_height / 2
    
    if x < mid_x and y < mid_y:
        return 1  # Top-Left
    elif x >= mid_x and y < mid_y:
        return 2  # Top-Right
    elif x < mid_x and y >= mid_y:
        return 3  # Bottom-Left
    else:
        return 4  # Bottom-Right


def analyze_ball_presence(events_path: str, tracks_path: str, frame_width: int, frame_height: int) -> Dict:
    """Analyze ball presence in each zone."""
    df = pd.read_csv(events_path)
    
    # Analyze ball positions from events
    zone_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    zone_frames = {1: [], 2: [], 3: [], 4: []}
    
    for _, row in df.iterrows():
        if pd.isna(row['pos_x']) or pd.isna(row['pos_y']):
            continue
        zone = get_zone(row['pos_x'], row['pos_y'], frame_width, frame_height)
        zone_counts[zone] += 1
        zone_frames[zone].append(int(row['frame']))
    
    # Also analyze from tracks if available
    if tracks_path and os.path.exists(tracks_path):
        try:
            with open(tracks_path, 'rb') as f:
                tracks = pickle.load(f)
            
            if 'ball' in tracks:
                for frame_idx, ball_frame in enumerate(tracks['ball']):
                    if 1 in ball_frame:
                        ball_info = ball_frame[1]
                        pos = ball_info.get('position') or ball_info.get('position_adjusted') or ball_info.get('center')
                        if pos and pos[0] is not None:
                            zone = get_zone(pos[0], pos[1], frame_width, frame_height)
                            zone_counts[zone] += 1
                            zone_frames[zone].append(frame_idx)
        except Exception as e:
            print(f"[WARN] Could not load tracks for ball analysis: {e}")
    
    total = sum(zone_counts.values())
    zone_percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in zone_counts.items()}
    
    return {
        'zone_counts': zone_counts,
        'zone_percentages': zone_percentages,
        'zone_frames': zone_frames,
        'total_samples': total,
    }


def analyze_player_density(tracks_path: str, frame_width: int, frame_height: int) -> Dict:
    """Analyze player density by team in each zone."""
    if not tracks_path or not os.path.exists(tracks_path):
        return {}
    
    zone_player_counts = {1: {'team1': 0, 'team2': 0}, 2: {'team1': 0, 'team2': 0},
                          3: {'team1': 0, 'team2': 0}, 4: {'team1': 0, 'team2': 0}}
    total_frames = 0
    
    try:
        with open(tracks_path, 'rb') as f:
            tracks = pickle.load(f)
        
        if 'players' not in tracks:
            return {}
        
        for frame_idx, players_frame in enumerate(tracks['players']):
            total_frames += 1
            for pid, pinfo in players_frame.items():
                team = pinfo.get('team', 0)
                if team not in (1, 2):
                    continue
                
                pos = pinfo.get('center_smoothed') or pinfo.get('center') or pinfo.get('position')
                if pos and pos[0] is not None:
                    zone = get_zone(pos[0], pos[1], frame_width, frame_height)
                    team_key = 'team1' if team == 1 else 'team2'
                    zone_player_counts[zone][team_key] += 1
    except Exception as e:
        print(f"[WARN] Could not analyze player density: {e}")
        return {}
    
    # Normalize by number of frames
    for zone in zone_player_counts:
        for team_key in zone_player_counts[zone]:
            zone_player_counts[zone][team_key] = zone_player_counts[zone][team_key] / total_frames if total_frames > 0 else 0
    
    return zone_player_counts


def create_ball_presence_heatmap(ball_data: Dict, frame_width: int, frame_height: int, output_path: str):
    """Create a heatmap showing ball presence in each zone."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create a grid representation
    grid = np.zeros((frame_height // 10, frame_width // 10))
    mid_x = frame_width // 2
    mid_y = frame_height // 2
    
    # Color each zone
    grid[:mid_y//10, :mid_x//10] = ball_data['zone_percentages'][1]  # Top-Left
    grid[:mid_y//10, mid_x//10:] = ball_data['zone_percentages'][2]  # Top-Right
    grid[mid_y//10:, :mid_x//10] = ball_data['zone_percentages'][3]  # Bottom-Left
    grid[mid_y//10:, mid_x//10:] = ball_data['zone_percentages'][4]  # Bottom-Right
    
    # Heatmap
    im1 = ax1.imshow(grid, cmap='YlOrRd', interpolation='bilinear', aspect='auto')
    ax1.set_title('Ball Presence Heatmap by Zone\n(Darker = More Ball Activity)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Pitch Width', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pitch Height', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Ball Activity (%)')
    
    # Add zone labels
    ax1.text(mid_x//20, mid_y//20, 'Zone 1\n(Top-Left)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    ax1.text(mid_x//20 + mid_x//10, mid_y//20, 'Zone 2\n(Top-Right)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    ax1.text(mid_x//20, mid_y//20 + mid_y//10, 'Zone 3\n(Bottom-Left)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
    ax1.text(mid_x//20 + mid_x//10, mid_y//20 + mid_y//10, 'Zone 4\n(Bottom-Right)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Bar chart
    zones = [f'Zone {i}\n({ZONE_NAMES[i]})' for i in [1, 2, 3, 4]]
    counts = [ball_data['zone_counts'][i] for i in [1, 2, 3, 4]]
    percentages = [ball_data['zone_percentages'][i] for i in [1, 2, 3, 4]]
    colors = ['red', 'green', 'blue', 'yellow']
    
    bars = ax2.bar(zones, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Ball Presence (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Ball Presence Percentage by Zone', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, max(percentages) * 1.2 if percentages else 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%\n({count} events)', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add descriptive text
    fig.text(0.5, 0.02, 
             'This visualization shows where the ball spends most time during the match. Zone 1-4 represent different areas of the pitch.', 
             ha='center', fontsize=10, style='italic', wrap=True)
    
    fig.suptitle('Ball Presence Analysis by Ground Zone', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[ZONE] Saved ball presence heatmap to {output_path}")
    plt.close()


def create_player_density_heatmap(player_data: Dict, frame_width: int, frame_height: int, output_path: str):
    """Create heatmaps showing player density by team in each zone."""
    if not player_data:
        print("[WARN] No player density data available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create grid for each visualization
    grid_size = (frame_height // 20, frame_width // 20)
    mid_x = frame_width // 2
    mid_y = frame_height // 2
    
    # Team 1 density
    grid_team1 = np.zeros(grid_size)
    grid_team1[:mid_y//20, :mid_x//20] = player_data[1]['team1']
    grid_team1[:mid_y//20, mid_x//20:] = player_data[2]['team1']
    grid_team1[mid_y//20:, :mid_x//20] = player_data[3]['team1']
    grid_team1[mid_y//20:, mid_x//20:] = player_data[4]['team1']
    
    im1 = axes[0].imshow(grid_team1, cmap='Blues', interpolation='bilinear', aspect='auto')
    axes[0].set_title('Team 1 Player Density\n(Blue = More Players)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Pitch Width', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Pitch Height', fontsize=10, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Avg Players per Frame')
    
    # Team 2 density
    grid_team2 = np.zeros(grid_size)
    grid_team2[:mid_y//20, :mid_x//20] = player_data[1]['team2']
    grid_team2[:mid_y//20, mid_x//20:] = player_data[2]['team2']
    grid_team2[mid_y//20:, :mid_x//20] = player_data[3]['team2']
    grid_team2[mid_y//20:, mid_x//20:] = player_data[4]['team2']
    
    im2 = axes[1].imshow(grid_team2, cmap='Reds', interpolation='bilinear', aspect='auto')
    axes[1].set_title('Team 2 Player Density\n(Red = More Players)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Pitch Width', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Pitch Height', fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Avg Players per Frame')
    
    # Combined comparison bar chart
    zones = ['Zone 1\n(Top-L)', 'Zone 2\n(Top-R)', 'Zone 3\n(Bot-L)', 'Zone 4\n(Bot-R)']
    team1_values = [player_data[i]['team1'] for i in [1, 2, 3, 4]]
    team2_values = [player_data[i]['team2'] for i in [1, 2, 3, 4]]
    
    x = np.arange(len(zones))
    width = 0.35
    axes[2].bar(x - width/2, team1_values, width, label='Team 1', color='#3498db', alpha=0.8)
    axes[2].bar(x + width/2, team2_values, width, label='Team 2', color='#e74c3c', alpha=0.8)
    axes[2].set_ylabel('Average Players per Frame', fontsize=11, fontweight='bold')
    axes[2].set_title('Player Density Comparison by Zone', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(zones)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (t1_val, t2_val) in enumerate(zip(team1_values, team2_values)):
        axes[2].text(i - width/2, t1_val + 0.01, f'{t1_val:.2f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[2].text(i + width/2, t2_val + 0.01, f'{t2_val:.2f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add descriptive text
    fig.text(0.5, 0.02, 
             'This visualization shows where each team positions their players most frequently. Higher values indicate more player presence in that zone.', 
             ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.suptitle('Player Density Analysis by Zone and Team', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[ZONE] Saved player density heatmap to {output_path}")
    plt.close()


def create_zone_overlay_video_frame(frame: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    """Overlay zone boundaries on a video frame."""
    overlay = frame.copy()
    mid_x = frame_width // 2
    mid_y = frame_height // 2
    
    # Draw zone boundaries
    cv2.line(overlay, (mid_x, 0), (mid_x, frame_height), (255, 255, 255), 2)
    cv2.line(overlay, (0, mid_y), (frame_width, mid_y), (255, 255, 255), 2)
    
    # Add zone labels with colored backgrounds
    zone_labels = [
        ((mid_x//4, mid_y//4), "Zone 1\nTop-Left", (255, 100, 100)),
        ((mid_x + mid_x//4, mid_y//4), "Zone 2\nTop-Right", (100, 255, 100)),
        ((mid_x//4, mid_y + mid_y//4), "Zone 3\nBottom-Left", (100, 100, 255)),
        ((mid_x + mid_x//4, mid_y + mid_y//4), "Zone 4\nBottom-Right", (255, 255, 100)),
    ]
    
    for (x, y), label, color in zone_labels:
        text_size = cv2.getTextSize(label.split('\n')[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, (x - 10, y - 25), (x + text_size[0] + 10, y + 5), color, -1)
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


def main():
    parser = argparse.ArgumentParser(description="Analyze ground zones and create visualizations")
    parser.add_argument("--events_path", type=str, default="output_videos/events.csv",
                        help="Path to events CSV")
    parser.add_argument("--tracks_path", type=str, default="stubs/track_stubs.pkl",
                        help="Path to tracks pickle file")
    parser.add_argument("--frame_width", type=int, default=1280, help="Video frame width")
    parser.add_argument("--frame_height", type=int, default=720, help="Video frame height")
    parser.add_argument("--output_dir", type=str, default="outputs/zone_analysis",
                        help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[ZONE] Analyzing ball presence...")
    ball_data = analyze_ball_presence(args.events_path, args.tracks_path, args.frame_width, args.frame_height)
    create_ball_presence_heatmap(ball_data, args.frame_width, args.frame_height,
                                os.path.join(args.output_dir, 'ball_presence_by_zone.png'))
    
    print("[ZONE] Analyzing player density...")
    player_data = analyze_player_density(args.tracks_path, args.frame_width, args.frame_height)
    if player_data:
        create_player_density_heatmap(player_data, args.frame_width, args.frame_height,
                                      os.path.join(args.output_dir, 'player_density_by_zone.png'))
    
    print(f"[ZONE] Analysis complete. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

