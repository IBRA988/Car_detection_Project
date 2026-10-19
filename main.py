"""
Production Command-Line Interface (CLI) for Traffic Vision Pro.
Supports configuration loading, parameter overrides, live video playback, and export.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from src.config import AppConfig, load_config
from src.pipeline import TrafficAnalysisPipeline

console = Console()


def setup_logger(verbose: bool = False) -> None:
    """Configure rich logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments with full flag support for pipeline tuning."""
    parser = argparse.ArgumentParser(
        description="Traffic Vision Pro: Real-time Multi-lane Vehicle Detection, Tracking, and Speed Estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/default_highway.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        default=None,
        help="Path to input video file (overrides config).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save output video with rendered telemetry (overrides config).",
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt) (overrides config).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold for vehicle detection (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference accelerator device ('auto', 'cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--calib",
        type=float,
        default=None,
        help="Pixel-to-meter calibration coefficient (overrides config).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display live video window with OpenCV GUI.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (useful for benchmarks/testing).",
    )
    parser.add_argument(
        "--export-stats",
        type=str,
        default=None,
        help="Export final traffic statistics to a JSON report file.",
    )
    parser.add_argument(
        "--show-lanes",
        action="store_true",
        default=False,
        help="Enable colored lane polygon shading on the road (disabled by default for a clean view).",
    )
    parser.add_argument(
        "--dual-line",
        action="store_true",
        default=False,
        help="Use dual boundary lines for counting instead of a single clean line.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logs.",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI execution flow."""
    args = parse_arguments()
    setup_logger(args.verbose)

    console.print(
        Panel.fit(
            "[bold cyan]Traffic Vision Pro[/bold cyan] - [dim]High-Throughput Vehicle Analytics[/dim]\n"
            "[green]Multi-Lane Tracking • Speed Estimation • Bilateral Traffic Gate Counting[/green]",
            border_style="cyan",
        )
    )

    # 1. Load base configuration
    config_path = args.config if os.path.exists(args.config) else None
    config: AppConfig = load_config(config_path)

    # 2. Apply CLI overrides
    if args.video:
        config.video.input_path = args.video
    if args.output is not None:
        config.video.output_path = args.output if args.output != "" else None
    if args.weights:
        config.model.weights = args.weights
    if args.conf is not None:
        config.model.confidence_threshold = args.conf
    if args.device:
        config.model.device = args.device
    if args.calib is not None:
        config.estimator.default_px_to_meter = args.calib
        for lane in config.lanes:
            object.__setattr__(lane, "px_to_meter", args.calib)
    if args.show_lanes:
        config.visualization.show_lanes = True
        config.visualization.show_lane_speeds = True
    if args.dual_line:
        config.visualization.single_line_gates = False

    # Validate inputs
    if not os.path.exists(config.video.input_path):
        console.print(f"[bold red]Error:[/bold red] Input video not found at: {config.video.input_path}")
        return 1

    if not os.path.exists(config.model.weights):
        console.print(f"[bold red]Error:[/bold red] Model weights not found at: {config.model.weights}")
        return 1

    # 3. Print Config Summary
    summary_table = Table(title="Pipeline Configuration", border_style="dim")
    summary_table.add_column("Parameter", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Input Video", config.video.input_path)
    summary_table.add_row("Output Video", str(config.video.output_path))
    summary_table.add_row("Weights", config.model.weights)
    summary_table.add_row("Confidence Threshold", f"{config.model.confidence_threshold:.2f}")
    summary_table.add_row("Device", config.model.device)
    summary_table.add_row("Calibration (px -> m)", f"{config.estimator.default_px_to_meter:.4f}")
    summary_table.add_row("Monitored Lanes", str(len(config.lanes)))
    summary_table.add_row("Counting Gates", str(len(config.counting_gates)))
    summary_table.add_row("Interactive Show", str(args.show))
    console.print(summary_table)

    # 4. Initialize and run pipeline
    pipeline = TrafficAnalysisPipeline(config)
    start_time = time.perf_counter()

    try:
        final_metrics = pipeline.run(
            show=args.show,
            max_frames=args.max_frames,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline execution interrupted by user.[/yellow]")
        final_metrics = pipeline.metrics
    except Exception as exc:
        console.print(f"[bold red]Pipeline failure:[/bold red] {exc}")
        logging.exception("Fatal error occurred during execution.")
        return 1

    total_time = time.perf_counter() - start_time
    avg_fps = final_metrics.frame_index / max(1e-4, total_time)

    # 5. Print Execution Report
    results_table = Table(title="Execution Telemetry & Results", border_style="green")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Result", style="bold yellow")

    results_table.add_row("Processed Frames", str(final_metrics.frame_index))
    results_table.add_row("Total Wall Time", f"{total_time:.2f} s")
    results_table.add_row("Average Throughput", f"{avg_fps:.1f} FPS")
    results_table.add_row("Left Gate Flow", str(final_metrics.count_left))
    results_table.add_row("Right Gate Flow", str(final_metrics.count_right))
    results_table.add_row("Total Counted Vehicles", str(final_metrics.total_counted))
    results_table.add_row("Active Tracks at Exit", str(final_metrics.active_tracks))
    console.print(results_table)

    # 6. Export stats report if requested
    if args.export_stats:
        stats_payload = {
            "processed_frames": final_metrics.frame_index,
            "total_wall_time_seconds": round(total_time, 2),
            "average_fps": round(avg_fps, 1),
            "total_vehicles_counted": final_metrics.total_counted,
            "flow_left": final_metrics.count_left,
            "flow_right": final_metrics.count_right,
            "class_breakdown": pipeline.counter.class_breakdown,
            "lane_speeds_kmh": [round(s, 1) for s in final_metrics.lane_speeds],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.export_stats)) or ".", exist_ok=True)
        with open(args.export_stats, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2)
        console.print(f"[green]Stats successfully exported to: {args.export_stats}[/green]")

    if config.video.output_path and os.path.exists(config.video.output_path):
        console.print(f"[bold green]Annotated video saved:[/bold green] {config.video.output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
