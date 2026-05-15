"""
Master Analysis Script

Runs all analysis modules in sequence:
1. Enhanced statistics generation
2. CSV visualizations
3. Zone analysis
4. Passing network (optional)
"""

import os
import sys
import subprocess
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"[ANALYSIS] {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=BASE_DIR, 
                              capture_output=False, text=True)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {description} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all analysis modules")
    parser.add_argument("--events_path", type=str, default="outputs/analysis/events.csv",
                        help="Path to events CSV")
    parser.add_argument("--tracks_path", type=str, default="stubs/track_stubs.pkl",
                        help="Path to tracks pickle file")
    parser.add_argument("--frame_width", type=int, default=1280,
                        help="Video frame width")
    parser.add_argument("--frame_height", type=int, default=720,
                        help="Video frame height")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Video FPS")
    parser.add_argument("--skip_network", action="store_true",
                        help="Skip passing network analysis")
    args = parser.parse_args()
    
    # Check if events file exists
    events_full_path = os.path.join(BASE_DIR, args.events_path)
    if not os.path.exists(events_full_path):
        print(f"[ERROR] Events CSV not found at {events_full_path}")
        print("[INFO] Please run main.py first to generate events.csv")
        return 1
    
    success_count = 0
    total_steps = 4 if not args.skip_network else 3
    
    # Step 1: Enhanced Statistics
    cmd1 = f'python analysis/enhanced_stats.py --events_path "{args.events_path}" --output_dir "outputs/analysis" --frame_width {args.frame_width} --frame_height {args.frame_height}'
    if run_command(cmd1, "Generating enhanced statistics"):
        success_count += 1
    
    # Step 2: CSV Visualizations
    stats_path = "outputs/analysis/comprehensive_stats_report.json"
    cmd2 = f'python analysis/visualize_csv.py --events_path "{args.events_path}" --stats_path "{stats_path}" --output_dir "outputs/visualizations" --fps {args.fps}'
    if run_command(cmd2, "Creating CSV visualizations"):
        success_count += 1
    
    # Step 3: Zone Analysis
    cmd3 = f'python analysis/zone_analysis.py --events_path "{args.events_path}" --tracks_path "{args.tracks_path}" --frame_width {args.frame_width} --frame_height {args.frame_height} --output_dir "outputs/zone_analysis"'
    if run_command(cmd3, "Analyzing ground zones"):
        success_count += 1
    
    # Step 4: Passing Network (optional)
    if not args.skip_network:
        cmd4 = f'python pass_network.py --events_path "{args.events_path}" --output_dir "outputs/pass_network_outputs" --period_strategy none --min_edge_count 3 --max_edges_to_plot 25 --node_label_top_k 5'
        if run_command(cmd4, "Generating passing network"):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"[SUMMARY] Completed {success_count}/{total_steps} analysis steps")
    print(f"{'='*60}\n")
    
    if success_count == total_steps:
        print("[SUCCESS] All analysis completed! Check outputs/ directory for results.")
        return 0
    else:
        print("[WARNING] Some analysis steps failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

