"""
Enhanced Statistics Generator

Generates comprehensive statistics from events.csv including:
- Player performance metrics
- Team statistics
- Pass completion rates
- Zone-based analysis
- Time-based trends
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


def load_events(events_path: str) -> pd.DataFrame:
    """Load and clean events CSV."""
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events CSV not found: {events_path}")
    df = pd.read_csv(events_path)
    df = df[df['type'] == 'pass'].copy()
    df = df.dropna(subset=['from_id', 'to_id'])
    df = df[df['from_id'] != df['to_id']]  # Remove self-passes
    return df


def compute_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive player statistics."""
    stats = []
    all_players = set(df['from_id'].unique()) | set(df['to_id'].unique())
    
    for pid in all_players:
        passes_made = df[df['from_id'] == pid]
        passes_received = df[df['to_id'] == pid]
        
        total_passes = len(passes_made)
        successful_passes = len(passes_made[~passes_made['interception']])
        interceptions_received = len(passes_received[passes_received['interception'] == True])
        
        completion_rate = (successful_passes / total_passes * 100) if total_passes > 0 else 0
        
        avg_pass_distance = passes_made['travel'].mean() if len(passes_made) > 0 else 0
        max_pass_distance = passes_made['travel'].max() if len(passes_made) > 0 else 0
        
        avg_ball_speed = passes_made['ball_speed'].mean() if len(passes_made) > 0 else 0
        
        # Team assignment (most common team)
        teams = passes_made['team'].value_counts()
        primary_team = teams.index[0] if len(teams) > 0 else 0
        
        stats.append({
            'player_id': int(pid),
            'team': int(primary_team),
            'passes_attempted': int(total_passes),
            'passes_completed': int(successful_passes),
            'passes_intercepted': int(interceptions_received),
            'completion_rate_%': round(completion_rate, 2),
            'avg_pass_distance_px': round(avg_pass_distance, 2),
            'max_pass_distance_px': round(max_pass_distance, 2),
            'avg_ball_speed_pxps': round(avg_ball_speed, 2),
            'passes_received': int(len(passes_received)),
        })
    
    return pd.DataFrame(stats).sort_values('passes_attempted', ascending=False)


def compute_team_stats(df: pd.DataFrame) -> Dict:
    """Compute team-level statistics."""
    team_stats = {}
    for team_id in [1, 2]:
        team_passes = df[df['team'] == team_id]
        total = len(team_passes)
        successful = len(team_passes[~team_passes['interception']])
        intercepted = len(team_passes[team_passes['interception'] == True])
        
        team_stats[f'team_{team_id}'] = {
            'total_passes': int(total),
            'successful_passes': int(successful),
            'intercepted_passes': int(intercepted),
            'success_rate_%': round((successful / total * 100) if total > 0 else 0, 2),
            'avg_pass_distance_px': round(team_passes['travel'].mean() if len(team_passes) > 0 else 0, 2),
            'unique_players': int(len(set(team_passes['from_id'].unique()) | set(team_passes['to_id'].unique()))),
        }
    
    return team_stats


def compute_zone_stats(df: pd.DataFrame, frame_width: int, frame_height: int) -> Dict:
    """Compute statistics for 4 ground zones."""
    # Divide ground into 4 zones: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    mid_x = frame_width / 2
    mid_y = frame_height / 2
    
    zones = {
        'zone_1_top_left': {'x_range': (0, mid_x), 'y_range': (0, mid_y), 'count': 0, 'passes': []},
        'zone_2_top_right': {'x_range': (mid_x, frame_width), 'y_range': (0, mid_y), 'count': 0, 'passes': []},
        'zone_3_bottom_left': {'x_range': (0, mid_x), 'y_range': (mid_y, frame_height), 'count': 0, 'passes': []},
        'zone_4_bottom_right': {'x_range': (mid_x, frame_width), 'y_range': (mid_y, frame_height), 'count': 0, 'passes': []},
    }
    
    for _, row in df.iterrows():
        x, y = row['pos_x'], row['pos_y']
        if pd.isna(x) or pd.isna(y):
            continue
        
        for zone_name, zone_data in zones.items():
            x_min, x_max = zone_data['x_range']
            y_min, y_max = zone_data['y_range']
            if x_min <= x < x_max and y_min <= y < y_max:
                zones[zone_name]['count'] += 1
                zones[zone_name]['passes'].append({
                    'frame': int(row['frame']),
                    'from_id': int(row['from_id']),
                    'to_id': int(row['to_id']),
                    'team': int(row['team']),
                    'interception': bool(row['interception']),
                })
                break
    
    # Convert to summary format
    zone_summary = {}
    for zone_name, zone_data in zones.items():
        zone_summary[zone_name] = {
            'total_passes': zone_data['count'],
            'successful_passes': sum(1 for p in zone_data['passes'] if not p['interception']),
            'intercepted_passes': sum(1 for p in zone_data['passes'] if p['interception']),
            'unique_players': len(set(p['from_id'] for p in zone_data['passes']) | set(p['to_id'] for p in zone_data['passes'])),
        }
    
    return zone_summary


def compute_time_based_stats(df: pd.DataFrame, fps: float = 25.0) -> Dict:
    """Compute statistics over time periods."""
    if len(df) == 0:
        return {}
    
    df['time_seconds'] = df['frame'] / fps
    total_duration = df['time_seconds'].max()
    
    # Divide into quarters
    quarter_duration = total_duration / 4
    quarters = {
        'Q1': {'start': 0, 'end': quarter_duration},
        'Q2': {'start': quarter_duration, 'end': 2 * quarter_duration},
        'Q3': {'start': 2 * quarter_duration, 'end': 3 * quarter_duration},
        'Q4': {'start': 3 * quarter_duration, 'end': total_duration},
    }
    
    quarter_stats = {}
    for q_name, q_range in quarters.items():
        q_passes = df[(df['time_seconds'] >= q_range['start']) & (df['time_seconds'] < q_range['end'])]
        quarter_stats[q_name] = {
            'total_passes': int(len(q_passes)),
            'successful_passes': int(len(q_passes[~q_passes['interception']])),
            'intercepted_passes': int(len(q_passes[q_passes['interception'] == True])),
            'avg_pass_distance_px': round(q_passes['travel'].mean() if len(q_passes) > 0 else 0, 2),
        }
    
    return quarter_stats


def generate_comprehensive_report(events_path: str, output_dir: str, frame_width: int = 1280, frame_height: int = 720):
    """Generate comprehensive statistics report."""
    print(f"[STATS] Loading events from {events_path}")
    df = load_events(events_path)
    
    if len(df) == 0:
        print("[WARN] No pass events found in CSV")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Player statistics
    print("[STATS] Computing player statistics...")
    player_stats = compute_player_stats(df)
    
    # Format CSV for better readability
    player_stats_formatted = player_stats.copy()
    player_stats_formatted['completion_rate_%'] = player_stats_formatted['completion_rate_%'].apply(lambda x: f"{x:.2f}%")
    player_stats_formatted['avg_pass_distance_px'] = player_stats_formatted['avg_pass_distance_px'].apply(lambda x: f"{x:.2f}")
    player_stats_formatted['max_pass_distance_px'] = player_stats_formatted['max_pass_distance_px'].apply(lambda x: f"{x:.2f}")
    player_stats_formatted['avg_ball_speed_pxps'] = player_stats_formatted['avg_ball_speed_pxps'].apply(lambda x: f"{x:.2f}")
    
    # Reorder columns for better readability
    column_order = ['player_id', 'team', 'passes_attempted', 'passes_completed', 'completion_rate_%', 
                    'passes_intercepted', 'passes_received', 'avg_pass_distance_px', 'max_pass_distance_px', 'avg_ball_speed_pxps']
    player_stats_formatted = player_stats_formatted[[col for col in column_order if col in player_stats_formatted.columns]]
    
    player_stats_path = os.path.join(output_dir, 'player_statistics.csv')
    player_stats_formatted.to_csv(player_stats_path, index=False)
    print(f"[STATS] Saved player statistics to {player_stats_path}")
    
    # Also save a summary CSV with top performers
    top_passers = player_stats.nlargest(10, 'passes_attempted')[['player_id', 'team', 'passes_attempted', 'passes_completed', 'completion_rate_%']]
    top_passers_path = os.path.join(output_dir, 'top_10_passers.csv')
    top_passers.to_csv(top_passers_path, index=False)
    print(f"[STATS] Saved top 10 passers to {top_passers_path}")
    
    # Team statistics
    print("[STATS] Computing team statistics...")
    team_stats = compute_team_stats(df)
    
    # Zone statistics
    print("[STATS] Computing zone statistics...")
    zone_stats = compute_zone_stats(df, frame_width, frame_height)
    
    # Time-based statistics
    print("[STATS] Computing time-based statistics...")
    time_stats = compute_time_based_stats(df)
    
    # Overall summary
    summary = {
        'total_passes': int(len(df)),
        'total_successful_passes': int(len(df[~df['interception']])),
        'total_interceptions': int(len(df[df['interception'] == True])),
        'overall_success_rate_%': round((len(df[~df['interception']]) / len(df) * 100) if len(df) > 0 else 0, 2),
        'unique_players': int(len(set(df['from_id'].unique()) | set(df['to_id'].unique()))),
        'avg_pass_distance_px': round(df['travel'].mean(), 2),
        'max_pass_distance_px': round(df['travel'].max(), 2),
        'avg_ball_speed_pxps': round(df['ball_speed'].mean(), 2),
        'frame_range': {'start': int(df['frame'].min()), 'end': int(df['frame'].max())},
        'team_statistics': team_stats,
        'zone_statistics': zone_stats,
        'time_period_statistics': time_stats,
    }
    
    # Save JSON report
    report_path = os.path.join(output_dir, 'comprehensive_stats_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[STATS] Saved comprehensive report to {report_path}")
    
    # Print summary
    print("\n=== COMPREHENSIVE STATISTICS SUMMARY ===")
    print(f"Total Passes: {summary['total_passes']}")
    print(f"Successful: {summary['total_successful_passes']} ({summary['overall_success_rate_%']}%)")
    print(f"Interceptions: {summary['total_interceptions']}")
    print(f"Unique Players: {summary['unique_players']}")
    print(f"Average Pass Distance: {summary['avg_pass_distance_px']:.2f} px")
    print("\n=== ZONE STATISTICS ===")
    for zone_name, zone_data in zone_stats.items():
        print(f"{zone_name}: {zone_data['total_passes']} passes ({zone_data['successful_passes']} successful)")
    print("========================================\n")


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced statistics from events CSV")
    parser.add_argument("--events_path", type=str, default="output_videos/events.csv",
                        help="Path to events CSV file")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis",
                        help="Output directory for statistics")
    parser.add_argument("--frame_width", type=int, default=1280,
                        help="Video frame width for zone calculations")
    parser.add_argument("--frame_height", type=int, default=720,
                        help="Video frame height for zone calculations")
    args = parser.parse_args()
    
    generate_comprehensive_report(args.events_path, args.output_dir, args.frame_width, args.frame_height)


if __name__ == "__main__":
    main()

