"""
CSV Data Visualization

Creates clean, self-explanatory visualizations from events.csv and statistics.
All charts include proper labels, legends, and titles for easy understanding.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List
import argparse

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set style for clean, professional plots
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False
    plt.style.use('default')
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


def load_data(events_path: str, stats_path: str = None) -> tuple:
    """Load events CSV and optional statistics JSON."""
    df = pd.read_csv(events_path)
    df = df[df['type'] == 'pass'].copy()
    
    stats = None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    
    return df, stats


def plot_pass_completion_by_player(df: pd.DataFrame, output_path: str):
    """Visualize pass completion rates for each player."""
    player_stats = []
    all_players = set(df['from_id'].unique()) | set(df['to_id'].unique())
    
    for pid in sorted(all_players):
        passes_made = df[df['from_id'] == pid]
        total = len(passes_made)
        successful = len(passes_made[~passes_made['interception']])
        completion = (successful / total * 100) if total > 0 else 0
        
        player_stats.append({
            'Player ID': f'P{pid}',
            'Passes Attempted': total,
            'Passes Completed': successful,
            'Completion Rate (%)': completion
        })
    
    stats_df = pd.DataFrame(player_stats)
    stats_df = stats_df[stats_df['Passes Attempted'] > 0].sort_values('Passes Attempted', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart: Passes attempted vs completed
    x = np.arange(len(stats_df))
    width = 0.35
    ax1.bar(x - width/2, stats_df['Passes Attempted'], width, label='Attempted', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, stats_df['Passes Completed'], width, label='Completed', color='#2ecc71', alpha=0.8)
    ax1.set_xlabel('Player ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Passes', fontsize=12, fontweight='bold')
    ax1.set_title('Passes Attempted vs Completed by Player', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['Player ID'], rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bar chart: Completion rate
    colors = ['#e74c3c' if rate < 50 else '#f39c12' if rate < 75 else '#2ecc71' 
              for rate in stats_df['Completion Rate (%)']]
    ax2.bar(stats_df['Player ID'], stats_df['Completion Rate (%)'], color=colors, alpha=0.8)
    ax2.axhline(y=75, color='green', linestyle='--', linewidth=2, label='Good (75%)')
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='Average (50%)')
    ax2.set_xlabel('Player ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Pass Completion Rate by Player', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticklabels(stats_df['Player ID'], rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Add overall title
    fig.suptitle('Player Pass Performance Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Add value labels on bars for better clarity
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        # Left chart - attempted
        ax1.text(i - width/2, row['Passes Attempted'] + 0.5, str(int(row['Passes Attempted'])), 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Left chart - completed
        ax1.text(i + width/2, row['Passes Completed'] + 0.5, str(int(row['Passes Completed'])), 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Right chart - completion rate
        ax2.text(i, row['Completion Rate (%)'] + 2, f'{row["Completion Rate (%)"]:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[VIZ] Saved pass completion chart to {output_path}")
    plt.close()


def plot_team_comparison(df: pd.DataFrame, output_path: str):
    """Compare team statistics side by side."""
    team_data = []
    for team_id in [1, 2]:
        team_passes = df[df['team'] == team_id]
        total = len(team_passes)
        successful = len(team_passes[~team_passes['interception']])
        intercepted = len(team_passes[team_passes['interception'] == True])
        
        team_data.append({
            'Team': f'Team {team_id}',
            'Total Passes': total,
            'Successful': successful,
            'Intercepted': intercepted,
            'Success Rate (%)': (successful / total * 100) if total > 0 else 0,
            'Avg Distance (px)': team_passes['travel'].mean() if len(team_passes) > 0 else 0,
        })
    
    team_df = pd.DataFrame(team_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total passes
    axes[0, 0].bar(team_df['Team'], team_df['Total Passes'], color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[0, 0].set_ylabel('Number of Passes', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Total Passes by Team', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(team_df['Total Passes']):
        axes[0, 0].text(i, v + 1, str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    # Success rate
    colors = ['#2ecc71' if rate > 70 else '#f39c12' if rate > 50 else '#e74c3c' 
              for rate in team_df['Success Rate (%)']]
    axes[0, 1].bar(team_df['Team'], team_df['Success Rate (%)'], color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Pass Success Rate by Team', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(team_df['Success Rate (%)']):
        axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Pass outcomes breakdown
    x = np.arange(len(team_df))
    width = 0.35
    axes[1, 0].bar(x - width/2, team_df['Successful'], width, label='Successful', color='#2ecc71', alpha=0.8)
    axes[1, 0].bar(x + width/2, team_df['Intercepted'], width, label='Intercepted', color='#e74c3c', alpha=0.8)
    axes[1, 0].set_ylabel('Number of Passes', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Pass Outcomes: Successful vs Intercepted', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(team_df['Team'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Average pass distance
    axes[1, 1].bar(team_df['Team'], team_df['Avg Distance (px)'], color=['#9b59b6', '#16a085'], alpha=0.8)
    axes[1, 1].set_ylabel('Average Distance (pixels)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Average Pass Distance by Team', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(team_df['Avg Distance (px)']):
        axes[1, 1].text(i, v + 2, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Add descriptive text box
    fig.text(0.5, 0.02, 'This chart compares key passing statistics between the two teams. Higher values indicate better performance.', 
             ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.suptitle('Team Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[VIZ] Saved team comparison chart to {output_path}")
    plt.close()


def plot_pass_distance_distribution(df: pd.DataFrame, output_path: str):
    """Visualize distribution of pass distances."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(df['travel'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(df['travel'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["travel"].mean():.1f}px')
    ax1.axvline(df['travel'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["travel"].median():.1f}px')
    ax1.set_xlabel('Pass Distance (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Pass Distances', fontsize=14, fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot by team
    team_data = [df[df['team'] == 1]['travel'].values, df[df['team'] == 2]['travel'].values]
    bp = ax2.boxplot(team_data, labels=['Team 1', 'Team 2'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
    ax2.set_ylabel('Pass Distance (pixels)', fontsize=12, fontweight='bold')
    ax2.set_title('Pass Distance Distribution by Team', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add descriptive annotations
    ax1.text(0.02, 0.98, f'Mean: {df["travel"].mean():.1f}px\nMedian: {df["travel"].median():.1f}px', 
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Pass Distance Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[VIZ] Saved pass distance distribution to {output_path}")
    plt.close()


def plot_time_series(df: pd.DataFrame, output_path: str, fps: float = 25.0):
    """Plot passes over time."""
    df['time_seconds'] = df['frame'] / fps
    df['successful'] = ~df['interception']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Passes over time
    time_bins = np.arange(0, df['time_seconds'].max() + 5, 5)  # 5-second bins
    successful_counts = []
    intercepted_counts = []
    bin_centers = []
    
    for i in range(len(time_bins) - 1):
        bin_data = df[(df['time_seconds'] >= time_bins[i]) & (df['time_seconds'] < time_bins[i+1])]
        successful_counts.append(len(bin_data[bin_data['successful']]))
        intercepted_counts.append(len(bin_data[~bin_data['successful']]))
        bin_centers.append((time_bins[i] + time_bins[i+1]) / 2)
    
    x = np.arange(len(bin_centers))
    width = 0.35
    axes[0].bar(x - width/2, successful_counts, width, label='Successful', color='#2ecc71', alpha=0.8)
    axes[0].bar(x + width/2, intercepted_counts, width, label='Intercepted', color='#e74c3c', alpha=0.8)
    axes[0].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Passes', fontsize=12, fontweight='bold')
    axes[0].set_title('Passes Over Time (5-second intervals)', fontsize=14, fontweight='bold', pad=20)
    axes[0].set_xticks(x[::2])
    axes[0].set_xticklabels([f'{int(t)}s' for t in bin_centers[::2]], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative passes
    df_sorted = df.sort_values('time_seconds')
    df_sorted['cumulative_successful'] = df_sorted['successful'].cumsum()
    df_sorted['cumulative_total'] = range(1, len(df_sorted) + 1)
    
    axes[1].plot(df_sorted['time_seconds'], df_sorted['cumulative_total'], 
                 label='Total Passes', color='#3498db', linewidth=2)
    axes[1].plot(df_sorted['time_seconds'], df_sorted['cumulative_successful'], 
                 label='Successful Passes', color='#2ecc71', linewidth=2)
    axes[1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Cumulative Passes', fontsize=12, fontweight='bold')
    axes[1].set_title('Cumulative Passes Over Time', fontsize=14, fontweight='bold', pad=20)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add summary statistics
    total_passes = len(df)
    successful_total = len(df[df['successful']])
    success_rate = (successful_total / total_passes * 100) if total_passes > 0 else 0
    
    fig.text(0.5, 0.02, 
             f'Match Summary: {total_passes} total passes | {successful_total} successful ({success_rate:.1f}%) | {total_passes - successful_total} intercepted', 
             ha='center', fontsize=10, fontweight='bold', style='italic')
    
    fig.suptitle('Pass Activity Over Time', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[VIZ] Saved time series chart to {output_path}")
    plt.close()


def create_all_visualizations(events_path: str, stats_path: str, output_dir: str, fps: float = 25.0):
    """Create all visualization charts."""
    print(f"[VIZ] Loading data from {events_path}")
    df, stats = load_data(events_path, stats_path)
    
    if len(df) == 0:
        print("[WARN] No pass events found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[VIZ] Generating visualizations...")
    plot_pass_completion_by_player(df, os.path.join(output_dir, '01_pass_completion_by_player.png'))
    plot_team_comparison(df, os.path.join(output_dir, '02_team_comparison.png'))
    plot_pass_distance_distribution(df, os.path.join(output_dir, '03_pass_distance_distribution.png'))
    plot_time_series(df, os.path.join(output_dir, '04_passes_over_time.png'), fps)
    
    print(f"[VIZ] All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create visualizations from events CSV")
    parser.add_argument("--events_path", type=str, default="output_videos/events.csv",
                        help="Path to events CSV")
    parser.add_argument("--stats_path", type=str, default="outputs/analysis/comprehensive_stats_report.json",
                        help="Path to statistics JSON (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations",
                        help="Output directory for charts")
    parser.add_argument("--fps", type=float, default=25.0, help="Video FPS for time calculations")
    args = parser.parse_args()
    
    create_all_visualizations(args.events_path, args.stats_path, args.output_dir, args.fps)


if __name__ == "__main__":
    main()

