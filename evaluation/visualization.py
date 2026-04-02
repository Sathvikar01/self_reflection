"""Visualization utilities for results."""

import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: tuple = (10, 6)
    dpi: int = 100
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


class ResultVisualizer:
    """Generate visualizations for experiment results.
    
    Note: This module provides configuration and data preparation.
    Actual rendering requires matplotlib/seaborn to be installed.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._data: Dict[str, Any] = {}
    
    def load_results(self, filepath: str) -> Dict:
        """Load results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        self._data = data
        return data
    
    def prepare_accuracy_comparison(
        self,
        baseline_acc: float,
        rl_acc: float,
    ) -> Dict[str, Any]:
        """Prepare data for accuracy bar chart."""
        return {
            "type": "bar",
            "data": {
                "labels": ["Baseline", "RL-Guided"],
                "values": [baseline_acc, rl_acc],
            },
            "config": {
                "title": "Accuracy Comparison",
                "ylabel": "Accuracy",
                "colors": self.config.color_palette[:2],
            },
        }
    
    def prepare_token_distribution(
        self,
        baseline_tokens: List[int],
        rl_tokens: List[int],
    ) -> Dict[str, Any]:
        """Prepare data for token distribution comparison."""
        return {
            "type": "histogram",
            "data": {
                "baseline": baseline_tokens,
                "rl": rl_tokens,
            },
            "config": {
                "title": "Token Distribution",
                "xlabel": "Tokens",
                "ylabel": "Frequency",
                "bins": 20,
            },
        }
    
    def prepare_backtrack_analysis(
        self,
        backtrack_counts: List[int],
        outcomes: List[bool],
    ) -> Dict[str, Any]:
        """Prepare data for backtrack effectiveness visualization."""
        successful = [c for c, o in zip(backtrack_counts, outcomes) if o]
        unsuccessful = [c for c, o in zip(backtrack_counts, outcomes) if not o]
        
        return {
            "type": "box",
            "data": {
                "successful": successful,
                "unsuccessful": unsuccessful,
            },
            "config": {
                "title": "Backtracks by Outcome",
                "ylabel": "Number of Backtracks",
            },
        }
    
    def prepare_score_trajectory(
        self,
        trajectories: List[List[float]],
    ) -> Dict[str, Any]:
        """Prepare data for score trajectory plot."""
        return {
            "type": "line",
            "data": {
                "trajectories": trajectories,
            },
            "config": {
                "title": "Score Trajectories",
                "xlabel": "Step",
                "ylabel": "Score",
            },
        }
    
    def prepare_action_distribution(
        self,
        expansions: int,
        reflections: int,
        backtracks: int,
    ) -> Dict[str, Any]:
        """Prepare data for action distribution pie chart."""
        return {
            "type": "pie",
            "data": {
                "labels": ["Expansions", "Reflections", "Backtracks"],
                "values": [expansions, reflections, backtracks],
            },
            "config": {
                "title": "Action Distribution",
                "colors": self.config.color_palette[:3],
            },
        }
    
    def prepare_summary_metrics(
        self,
        baseline_stats: Dict,
        rl_stats: Dict,
    ) -> Dict[str, Any]:
        """Prepare summary metrics comparison."""
        metrics = ["accuracy", "avg_score", "avg_tokens", "avg_latency"]
        
        return {
            "type": "table",
            "data": {
                "headers": ["Metric", "Baseline", "RL", "Improvement"],
                "rows": [
                    [
                        m.replace("_", " ").title(),
                        f"{baseline_stats.get(m, 0):.3f}",
                        f"{rl_stats.get(m, 0):.3f}",
                        f"{rl_stats.get(m, 0) - baseline_stats.get(m, 0):+.3f}",
                    ]
                    for m in metrics
                ],
            },
        }
    
    def export_plot_data(self, output_dir: str) -> None:
        """Export all plot data to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, data in self._data.items():
            filepath = output_path / f"{name}.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)


def generate_latex_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
) -> str:
    """Generate LaTeX table from data.
    
    Args:
        headers: Table headers
        rows: Table rows
        caption: Table caption
        label: Table label for references
    
    Returns:
        LaTeX table string
    """
    num_cols = len(headers)
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\begin{{tabular}}{{{'l' * num_cols}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])
    
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_paper_figure_code(
    plot_type: str,
    data: Dict[str, Any],
    filename: str,
) -> str:
    """Generate Python code for creating paper figures.
    
    Args:
        plot_type: Type of plot
        data: Plot data
        filename: Output filename
    
    Returns:
        Python code string
    """
    code_templates = {
        "bar": f'''
import matplotlib.pyplot as plt
import numpy as np

labels = {data.get("data", {}).get("labels", [])}
values = {data.get("data", {}).get("values", [])}

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
ax.set_ylabel('Accuracy')
ax.set_title('{data.get("config", {}).get("title", "")}')
ax.set_ylim(0, 1)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{{val:.1%}}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('{filename}', dpi=300)
plt.close()
''',
        "line": f'''
import matplotlib.pyplot as plt
import numpy as np

trajectories = {data.get("data", {}).get("trajectories", [])}

fig, ax = plt.subplots(figsize=(10, 6))
for i, traj in enumerate(trajectories):
    ax.plot(traj, alpha=0.7, label=f'Trajectory {{i+1}}')

ax.set_xlabel('Step')
ax.set_ylabel('Score')
ax.set_title('{data.get("config", {}).get("title", "")}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('{filename}', dpi=300)
plt.close()
''',
    }
    
    return code_templates.get(plot_type, f"# Unsupported plot type: {plot_type}")


class FigureGenerator:
    """Generate figures for paper."""
    
    def __init__(self, output_dir: str = "paper/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_figures(
        self,
        baseline_results: Dict,
        rl_results: Dict,
    ) -> List[str]:
        """Generate all paper figures.
        
        Args:
            baseline_results: Baseline experiment results
            rl_results: RL experiment results
        
        Returns:
            List of generated file paths
        """
        generated = []
        
        return generated
    
    def _generate_accuracy_figure(
        self,
        baseline_acc: float,
        rl_acc: float,
    ) -> str:
        """Generate accuracy comparison figure."""
        data = {
            "data": {
                "labels": ["Baseline", "RL-Guided"],
                "values": [baseline_acc, rl_acc],
            },
            "config": {"title": "Accuracy Comparison"},
        }
        
        code = generate_paper_figure_code("bar", data, str(self.output_dir / "accuracy.png"))
        
        code_path = self.output_dir / "generate_accuracy.py"
        with open(code_path, "w") as f:
            f.write(code)
        
        return str(code_path)
