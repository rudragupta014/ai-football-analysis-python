"""
pass_network.py

Builds passing networks and centrality reports from output_videos/events.csv.
This script does NOT modify the main pipeline; it only reads existing artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None


plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class PeriodSlice:
    label: str
    start_frame: int
    end_frame: int

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


def _log(msg: str):
    print(f"[pass_network] {msg}")


def parse_args():
    parser = argparse.ArgumentParser(description="Passing network + centrality analysis")
    parser.add_argument("--events_path", type=str, default="output_videos/events.csv",
                        help="Events CSV produced by main pipeline.")
    parser.add_argument("--player_meta", type=str, default="",
                        help="Optional CSV with columns [player_id,name,role,team].")
    parser.add_argument("--team_id", type=int, default=0,
                        help="Filter to a single team (0 = both).")
    parser.add_argument("--window_size", type=int, default=0,
                        help="Sliding-window size (frames by default). Set with --window_unit.")
    parser.add_argument("--window_unit", type=str, choices=["frames", "seconds"], default="frames",
                        help="Interpretation of --window_size.")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="FPS needed if window_unit=seconds.")
    parser.add_argument("--min_edge_count", type=int, default=2,
                        help="Ignore pass edges with fewer than N completed passes.")
    parser.add_argument("--normalize_by_possession", action="store_true",
                        help="Normalize edge weights by total passes in the slice.")
    parser.add_argument("--meters_per_pixel", type=float, default=0.0,
                        help="Multiply travel distance by this to convert to meters when "
                             "position_transformed is unavailable.")
    parser.add_argument("--min_pass_travel", type=float, default=2.0,
                        help="Minimum ball travel to consider a pass (after scaling).")
    parser.add_argument("--max_edges_to_plot", type=int, default=40,
                        help="Limit plotted edges to avoid clutter.")
    parser.add_argument("--node_label_top_k", type=int, default=6,
                        help="Show labels only for top-k players by pagerank.")
    parser.add_argument("--edge_width_scale", type=float, default=1.5,
                        help="Multiplier for edge widths after log scaling.")
    parser.add_argument("--output_dir", type=str, default="pass_network_outputs",
                        help="Where to store adjacency matrices, plots, and reports.")
    parser.add_argument("--plot_format", type=str, choices=["png", "svg"], default="png")
    parser.add_argument("--interactive", action="store_true",
                        help="Generate an optional Plotly HTML network (requires plotly).")
    parser.add_argument("--min_pass_threshold", type=int, default=5,
                        help="Sanity check threshold for reporting top passers.")
    parser.add_argument("--period_strategy", type=str, choices=["halves", "thirds", "none"],
                        default="none", help="Additional coarse slicing besides sliding windows.")
    parser.add_argument("--window_stride", type=int, default=0,
                        help="Stride for sliding windows (default = window_size).")
    return parser.parse_args()


def load_events(path: str, meters_per_pixel: float, min_pass_travel: float) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Events CSV not found at {path}")
    df = pd.read_csv(path)
    required = {"frame", "type", "from_id", "to_id", "team"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in events CSV: {missing}")

    df = df[df["type"].str.lower() == "pass"].copy()
    df = df.dropna(subset=["from_id", "to_id"])
    df = df[df["from_id"] != df["to_id"]]
    df["interception"] = df.get("interception", False).astype(bool)
    df["travel"] = df.get("travel", np.nan)
    df["ball_speed"] = df.get("ball_speed", np.nan)
    df["pos_x"] = df.get("pos_x", np.nan)
    df["pos_y"] = df.get("pos_y", np.nan)
    df["frame"] = df["frame"].astype(int)
    if meters_per_pixel and not df["travel"].isna().all():
        df["pass_length"] = df["travel"].fillna(0) * meters_per_pixel
        df["length_units"] = "m"
    else:
        df["pass_length"] = df["travel"].fillna(df["ball_speed"].fillna(0))
        df["length_units"] = "px"
    df = df[df["pass_length"].abs() >= min_pass_travel]
    return df


def load_player_meta(path: str) -> Dict[int, Dict[str, str]]:
    if not path or not os.path.exists(path):
        return {}
    meta_df = pd.read_csv(path)
    if "player_id" not in meta_df.columns:
        raise ValueError("player_meta CSV must contain 'player_id'")
    meta = {}
    for _, row in meta_df.iterrows():
        pid = int(row["player_id"])
        meta[pid] = {
            "name": row.get("name", f"Player {pid}"),
            "role": row.get("role", "Unknown"),
            "team": row.get("team", ""),
        }
    return meta


def derive_periods(df: pd.DataFrame, strategy: str, window_size: int,
                   window_unit: str, fps: float, window_stride: int) -> List[PeriodSlice]:
    periods = [PeriodSlice("full_match", df["frame"].min(), df["frame"].max())]
    max_frame = df["frame"].max()
    min_frame = df["frame"].min()

    if strategy == "halves":
        mid = (max_frame + min_frame) // 2
        periods.append(PeriodSlice("first_half", min_frame, mid))
        periods.append(PeriodSlice("second_half", mid + 1, max_frame))
    elif strategy == "thirds":
        span = max_frame - min_frame + 1
        chunk = span // 3
        periods.append(PeriodSlice("segment_1", min_frame, min_frame + chunk))
        periods.append(PeriodSlice("segment_2", min_frame + chunk + 1, min_frame + 2 * chunk))
        periods.append(PeriodSlice("segment_3", min_frame + 2 * chunk + 1, max_frame))

    if window_size and window_size > 0:
        stride = window_stride if window_stride and window_stride > 0 else window_size
        frame_window = window_size if window_unit == "frames" else int(window_size * fps)
        frame_stride = stride if window_unit == "frames" else int(stride * fps)
        for start in range(min_frame, max_frame, frame_stride):
            end = min(max_frame, start + frame_window)
            label = f"window_{start}_{end}"
            periods.append(PeriodSlice(label, start, end))
            if end >= max_frame:
                break

    return periods


def filter_events(df: pd.DataFrame, team_id: int, period: PeriodSlice) -> pd.DataFrame:
    subset = df[(df["frame"] >= period.start_frame) & (df["frame"] <= period.end_frame)].copy()
    if team_id:
        subset = subset[subset["team"] == team_id]
    return subset


def build_adj_matrix(df: pd.DataFrame, normalize: bool, max_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame()
    counts = df.groupby(["from_id", "to_id"]).agg(
        count=("frame", "size"),
        avg_length=("pass_length", "mean"),
        completion_rate=("interception", lambda x: 1 - x.mean())
    ).reset_index()
    if normalize:
        total = counts["count"].sum()
        counts["weight"] = counts["count"] / total if total > 0 else 0
    else:
        counts["weight"] = counts["count"]

    counts = counts.sort_values("count", ascending=False).head(max_edges)

    players = sorted(set(counts["from_id"]).union(counts["to_id"]))
    matrix = pd.DataFrame(0.0, index=players, columns=players)
    for _, row in counts.iterrows():
        matrix.loc[row["from_id"], row["to_id"]] = row["weight"]
    return matrix, counts


def compute_centrality(graph: nx.DiGraph) -> Dict[int, Dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    centrality = {}
    for node in graph.nodes():
        centrality[node] = {}

    in_degree = dict(graph.in_degree(weight="weight"))
    out_degree = dict(graph.out_degree(weight="weight"))
    total_degree = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    try:
        eigen = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        eigen = {n: 0.0 for n in graph.nodes()}
    pagerank = nx.pagerank(graph, weight="weight")
    clustering = nx.clustering(graph.to_undirected(), weight="weight")

    for node in graph.nodes():
        centrality[node].update({
            "degree": total_degree.get(node, 0.0),
            "in_degree": in_degree.get(node, 0.0),
            "out_degree": out_degree.get(node, 0.0),
            "betweenness": betweenness.get(node, 0.0),
            "eigenvector": eigen.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0),
            "clustering": clustering.get(node, 0.0),
        })
    return centrality


def summarize_players(df: pd.DataFrame, centrality: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    attempts = df.groupby("from_id").size()
    completions = df[~df["interception"]].groupby("from_id").size()
    avg_length = df.groupby("from_id")["pass_length"].mean()

    summary = {}
    for pid, metrics in centrality.items():
        att = float(attempts.get(pid, 0))
        comp = float(completions.get(pid, 0))
        completion_rate = (comp / att) if att > 0 else 0.0
        summary[pid] = {
            **metrics,
            "passes_attempted": att,
            "passes_completed": comp,
            "pass_completion_rate": completion_rate,
            "avg_pass_length": float(avg_length.get(pid, 0.0)),
        }
    return summary


def compute_player_positions(df: pd.DataFrame, meters_per_pixel: float) -> Dict[int, Tuple[float, float]]:
    pos_map = {}
    if {"pos_x", "pos_y"}.issubset(df.columns):
        grouped = df.groupby("from_id")[["pos_x", "pos_y"]].mean()
        if meters_per_pixel and meters_per_pixel > 0:
            grouped = grouped * meters_per_pixel
        for pid, row in grouped.iterrows():
            pos_map[int(pid)] = (float(row["pos_x"]), float(row["pos_y"]))
    return pos_map


def normalize_positions(pos_map: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    if not pos_map:
        return {}
    xs = [p[0] for p in pos_map.values()]
    ys = [p[1] for p in pos_map.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    scaled = {}
    for pid, (x, y) in pos_map.items():
        scaled[pid] = ((x - min_x) / span_x, 1 - (y - min_y) / span_y)
    return scaled


def plot_adj_heatmap(matrix: pd.DataFrame, team_label: str, period_label: str, output_dir: str, fmt: str):
    if matrix.empty:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix.values, cmap="Greens")
    plt.colorbar(label="Weight")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    plt.title(f"Adjacency Heatmap – {team_label} – {period_label}")
    plt.tight_layout()
    path = os.path.join(output_dir, f"adjacency_heatmap_{team_label}_{period_label}.{fmt}")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pass_density(df: pd.DataFrame, team_label: str, period_label: str, output_dir: str, fmt: str):
    if {"pos_x", "pos_y"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        plt.hist2d(df["pos_x"], df["pos_y"], bins=40, cmap="YlGnBu")
        plt.colorbar(label="Pass Count")
        plt.title(f"Pass Origin Density – {team_label} – {period_label}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        path = os.path.join(output_dir, f"pass_density_{team_label}_{period_label}.{fmt}")
        plt.savefig(path, dpi=200)
        plt.close()


def plot_clean_network(graph: nx.DiGraph,
                       counts_df: pd.DataFrame,
                       team_label: str,
                       period_label: str,
                       output_dir: str,
                       fmt: str,
                       player_meta: Dict[int, Dict[str, str]],
                       centrality: Dict[int, Dict[str, float]],
                       pos_map: Dict[int, Tuple[float, float]],
                       node_label_top_k: int,
                       edge_width_scale: float,
                       interactive: bool):
    if graph.number_of_edges() == 0:
        _log(f"No edges to plot for {team_label} / {period_label}")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    base_layout = normalize_positions(pos_map) if pos_map else {}
    if len(base_layout) < graph.number_of_nodes():
        fallback = nx.spring_layout(graph, seed=42, weight="weight")
        fallback.update(base_layout)
        layout = fallback
    else:
        layout = base_layout
    node_sizes = []
    for n in graph.nodes():
        degree = centrality[n]["out_degree"]
        node_sizes.append(600 + 2500 * (degree / max(1e-5, max(centrality[k]["out_degree"] for k in graph.nodes()))))
    edge_widths = []
    edge_alpha = []
    for u, v in graph.edges():
        w = graph[u][v]["weight"]
        scaled = math.log1p(max(w, 0.001)) * edge_width_scale
        edge_widths.append(scaled)
        edge_alpha.append(min(0.85, 0.3 + 0.15 * math.log1p(w)))

    labels = {}
    for node in graph.nodes():
        meta = player_meta.get(node, {})
        labels[node] = meta.get("name", str(node))

    for idx, (u, v) in enumerate(graph.edges()):
        nx.draw_networkx_edges(graph, layout, edgelist=[(u, v)], width=edge_widths[idx],
                               alpha=edge_alpha[idx], arrows=True, arrowstyle="-|>")
    nx.draw_networkx_nodes(graph, layout, node_size=node_sizes, node_color="#69c0ff", edgecolors="black")
    top_nodes = sorted(graph.nodes(), key=lambda n: centrality[n]["pagerank"], reverse=True)[:node_label_top_k]
    label_subset = {n: labels[n] for n in top_nodes}
    nx.draw_networkx_labels(graph, layout, labels=label_subset, font_size=10)
    plt.title(f"Passing Network – {team_label} – {period_label}")
    plt.axis("off")
    out_path = os.path.join(output_dir, f"pass_network_{team_label}_{period_label}.{fmt}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    _log(f"Saved network plot: {out_path}")

    if interactive and go is not None:
        edge_x = []
        edge_y = []
        for u, v in graph.edges():
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#aaa"),
                                hoverinfo="none", mode="lines")
        node_x = [layout[n][0] for n in graph.nodes()]
        node_y = [layout[n][1] for n in graph.nodes()]
        hover = [f"{labels[n]}<br>Degree: {centrality[n]['degree']:.2f}" for n in graph.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=list(labels.values()),
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                color=[centrality[n]["pagerank"] for n in graph.nodes()],
                size=[max(12, c * 80 + 20) for c in edge_widths[:len(graph.nodes())]],
                line=dict(color="black", width=0.5),
                colorbar=dict(title="Pagerank"))
        )
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f"Passing Network – {team_label} – {period_label}",
                            showlegend=False,
                            hovermode="closest"
                        ))
        html_path = os.path.join(output_dir, f"pass_network_{team_label}_{period_label}.html")
        fig.write_html(html_path)
        _log(f"Saved interactive network: {html_path}")


def run_sanity_checks(counts_df: pd.DataFrame, min_pass_threshold: int):
    if counts_df.empty:
        _log("WARNING: No pass edges after filtering; check filters.")
        return
    top_passers = counts_df.groupby("from_id")["count"].sum().sort_values(ascending=False).head(5)
    _log("Top passers (by completed count):")
    for pid, cnt in top_passers.items():
        status = "OK" if cnt >= min_pass_threshold else "LOW"
        _log(f"  Player {pid}: {cnt} passes [{status}]")
    connectivity = counts_df.shape[0]
    if connectivity == 0:
        _log("WARNING: Graph has zero edges; centrality metrics will be zero.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    events = load_events(args.events_path, args.meters_per_pixel, args.min_pass_travel)
    player_meta = load_player_meta(args.player_meta)
    periods = derive_periods(events, args.period_strategy, args.window_size,
                             args.window_unit, args.fps, args.window_stride)

    report = []
    edges_accumulator = []

    teams = [args.team_id] if args.team_id else sorted(events["team"].dropna().unique())
    for team in teams:
        team_label = f"team{team}" if team else "all_teams"
        for period in periods:
            sliced = filter_events(events, team, period)
            if sliced.empty:
                continue
            matrix, counts_df = build_adj_matrix(sliced, args.normalize_by_possession, args.max_edges_to_plot)
            if matrix.empty:
                continue

            counts_df = counts_df[counts_df["count"] >= args.min_edge_count]
            if counts_df.empty:
                continue

            run_sanity_checks(counts_df, args.min_pass_threshold)

            adjacency_path = os.path.join(
                args.output_dir, f"adjacency_{team_label}_{period.label}.csv")
            matrix.to_csv(adjacency_path)
            plot_adj_heatmap(matrix, team_label, period.label, args.output_dir, args.plot_format)
            plot_pass_density(sliced, team_label, period.label, args.output_dir, args.plot_format)

            G = nx.DiGraph()
            for _, row in counts_df.iterrows():
                G.add_edge(
                    int(row["from_id"]),
                    int(row["to_id"]),
                    weight=float(row["weight"]),
                    count=int(row["count"]),
                    avg_length=float(row["avg_length"]),
                    completion_rate=float(row["completion_rate"]),
                )

            centrality = compute_centrality(G)
            player_summary = summarize_players(sliced, centrality)
            summary_df = pd.DataFrame.from_dict(player_summary, orient='index').reset_index().rename(columns={'index': 'player_id'})
            summary_path = os.path.join(args.output_dir, f"player_summary_{team_label}_{period.label}.csv")
            summary_df.to_csv(summary_path, index=False)

            report.append({
                "team": int(team) if team else 0,
                "period": period.label,
                "players": player_summary,
                "player_summary_path": summary_path,
                "top_edges": counts_df.sort_values("count", ascending=False).head(10).to_dict(orient="records")
            })

            counts_df["team"] = team
            counts_df["period"] = period.label
            edges_accumulator.append(counts_df)

            plot_dir = os.path.join(args.output_dir, "plots")
            pos_map = compute_player_positions(sliced, args.meters_per_pixel)
            plot_clean_network(G, counts_df, team_label, period.label, plot_dir,
                               args.plot_format, player_meta, centrality, pos_map,
                               args.node_label_top_k, args.edge_width_scale, args.interactive)

    report_path = os.path.join(args.output_dir, "pass_network_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _log(f"Wrote report: {report_path}")

    if edges_accumulator:
        edges_df = pd.concat(edges_accumulator, ignore_index=True)
        edges_path = os.path.join(args.output_dir, "pass_network_edges.csv")
        edges_df.to_csv(edges_path, index=False)
        _log(f"Wrote edge list: {edges_path}")


if __name__ == "__main__":
    main()

