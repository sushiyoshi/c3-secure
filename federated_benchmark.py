"""Benchmark plaintext, TFHE, and CKKS federated aggregation strategies."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime when plotting is required
    plt = None  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from aggregators import (
    BaseAggregator,
    CKKSModelAggregator,
    PlaintextAggregator,
    TFHEModelAggregator,
    clone_state_dict,
)


class CustomDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 3, output_size: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32))
            self.fc.bias.copy_(torch.tensor([0.1, 0.2], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def create_simple_data(
    batch_size: int,
    num_samples: int,
    seed: Optional[int] = None,
) -> DataLoader:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_samples, 3)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] - features[:, 2] > 0).astype(np.int64)
    dataset = CustomDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Client:
    def __init__(
        self,
        client_id: int,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 0.01,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.model = NeuralNetwork().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def local_update(self, global_weights: OrderedDict[str, torch.Tensor], epochs: int) -> OrderedDict:
        self.model.load_state_dict(global_weights)
        self.model.train()
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return clone_state_dict(self.model.state_dict())

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        return (correct / total) * 100.0 if total else 0.0


class FederatedServer:
    def __init__(
        self,
        num_clients: int,
        aggregation_mode: str,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = NeuralNetwork().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.num_clients = num_clients
        self.aggregation_mode = aggregation_mode
        self.aggregator = self._create_aggregator()

    def _create_aggregator(self) -> BaseAggregator:
        if self.aggregation_mode == "plaintext":
            return PlaintextAggregator(self.global_model, self.num_clients)
        if self.aggregation_mode == "tfhe":
            return TFHEModelAggregator(self.global_model, self.num_clients)
        if self.aggregation_mode == "ckks":
            return CKKSModelAggregator(self.global_model, self.num_clients)
        raise ValueError(f"Unsupported aggregation mode: {self.aggregation_mode}")

    def evaluate(self, dataloader: DataLoader) -> float:
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        return (correct / total) * 100.0 if total else 0.0

    def aggregate(self, client_weights: Iterable[OrderedDict]) -> None:
        aggregated_state = self.aggregator.aggregate(client_weights)
        self.global_model.load_state_dict(aggregated_state)

    def get_global_weights(self) -> OrderedDict:
        return clone_state_dict(self.global_model.state_dict())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA is optional in CI
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    aggregation_mode: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    base_seed: int,
) -> Dict[str, List[float]]:
    seed_everything(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders = [
        create_simple_data(batch_size=16, num_samples=100, seed=base_seed + client_id * 17)
        for client_id in range(num_clients)
    ]
    test_loaders = [
        create_simple_data(batch_size=32, num_samples=50, seed=base_seed + 1000 + client_id * 19)
        for client_id in range(num_clients)
    ]

    server = FederatedServer(num_clients=num_clients, aggregation_mode=aggregation_mode, device=device)
    global_test_loader = create_simple_data(batch_size=64, num_samples=200, seed=base_seed + 999)
    clients = [
        Client(
            client_id=i + 1,
            device=device,
            train_loader=train_loaders[i],
            test_loader=test_loaders[i],
        )
        for i in range(num_clients)
    ]

    accuracy_history: List[float] = []
    round_time_history: List[float] = []

    for round_index in range(num_rounds):
        round_start = time.perf_counter()
        global_weights = server.get_global_weights()

        client_updates = [
            client.local_update(global_weights, epochs=local_epochs)
            for client in clients
        ]

        server.aggregate(client_updates)

        accuracy = server.evaluate(global_test_loader)
        round_duration = time.perf_counter() - round_start

        accuracy_history.append(accuracy)
        round_time_history.append(round_duration)

    return {
        "accuracy": accuracy_history,
        "round_time": round_time_history,
    }


def collect_metrics(
    aggregation_mode: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    repetitions: int,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    accuracy_runs = []
    round_time_runs = []

    for rep in range(repetitions):
        seed = base_seed + rep * 100
        results = run_single_experiment(
            aggregation_mode=aggregation_mode,
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            base_seed=seed,
        )
        accuracy_runs.append(results["accuracy"])
        round_time_runs.append(results["round_time"])

    accuracy_array = np.array(accuracy_runs)
    time_array = np.array(round_time_runs)

    return {
        "accuracy_mean": accuracy_array.mean(axis=0),
        "accuracy_std": accuracy_array.std(axis=0),
        "round_time_mean": time_array.mean(axis=0),
        "round_time_std": time_array.std(axis=0),
    }


COLOR_MAP = {
    "plaintext": "#1f77b4",
    "tfhe": "#ff7f0e",
    "ckks": "#2ca02c",
}


def _plot_results_svg(
    results: Dict[str, Dict[str, np.ndarray]],
    combos: Sequence[Sequence[str]],
    output_dir: Path,
    num_rounds: int,
) -> List[Path]:
    saved_paths: List[Path] = []
    round_positions = [i / max(1, num_rounds - 1) for i in range(num_rounds)]

    for combo in combos:
        width, height = 1200, 440
        margin_x, margin_y = 80, 70
        panel_gap = 120
        panel_width = (width - 2 * margin_x - panel_gap) / 2
        panel_height = height - 2 * margin_y

        def gather_bounds(key_mean: str, key_std: str) -> Tuple[float, float]:
            values: List[float] = []
            for mode in combo:
                metrics = results[mode]
                mean_vals = metrics[key_mean]
                std_vals = metrics[key_std]
                values.extend((mean_vals - std_vals).tolist())
                values.extend((mean_vals + std_vals).tolist())
            min_val = float(min(values))
            max_val = float(max(values))
            if math.isclose(min_val, max_val):
                max_val = min_val + 1.0
            return min_val, max_val

        accuracy_bounds = gather_bounds("accuracy_mean", "accuracy_std")
        time_bounds = gather_bounds("round_time_mean", "round_time_std")

        svg_lines: List[str] = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "  <style>text{font-family:Arial,sans-serif;font-size:18px;} .legend{font-size:20px;font-weight:bold;}</style>",
            "  <rect width='100%' height='100%' fill='white'/>",
            "  <text x='{0}' y='{1}' class='legend'>実験結果</text>".format(width / 2, 40),
        ]

        def add_panel(
            origin_x: float,
            origin_y: float,
            bounds: Tuple[float, float],
            metric_mean: str,
            metric_std: str,
            title: str,
            y_label: str,
        ) -> None:
            min_val, max_val = bounds
            svg_lines.append(
                "  <rect x='{:.2f}' y='{:.2f}' width='{:.2f}' height='{:.2f}' fill='none' stroke='#444' stroke-width='2'/>".format(
                    origin_x,
                    origin_y,
                    panel_width,
                    panel_height,
                )
            )
            svg_lines.append(
                "  <text x='{:.2f}' y='{:.2f}' text-anchor='middle' class='legend'>{}</text>".format(
                    origin_x + panel_width / 2,
                    origin_y - 20,
                    title,
                )
            )

            tick_values = np.linspace(min_val, max_val, num=5)
            for tick in tick_values:
                ratio = (tick - min_val) / (max_val - min_val)
                y_pos = origin_y + panel_height - ratio * panel_height
                svg_lines.append(
                    "  <line x1='{:.2f}' y1='{:.2f}' x2='{:.2f}' y2='{:.2f}' stroke='#ddd' stroke-width='1'/>".format(
                        origin_x,
                        y_pos,
                        origin_x + panel_width,
                        y_pos,
                    )
                )
                svg_lines.append(
                    "  <text x='{:.2f}' y='{:.2f}' text-anchor='end'>{:.2f}</text>".format(
                        origin_x - 10,
                        y_pos + 5,
                        tick,
                    )
                )

            svg_lines.append(
                "  <text x='{:.2f}' y='{:.2f}' text-anchor='middle'>{}</text>".format(
                    origin_x + panel_width / 2,
                    origin_y + panel_height + 40,
                    "Learning Round",
                )
            )
            svg_lines.append(
                "  <text x='{:.2f}' y='{:.2f}' text-anchor='middle' transform='rotate(-90,{:.2f},{:.2f})'>{}</text>".format(
                    origin_x - 70,
                    origin_y + panel_height / 2,
                    origin_x - 70,
                    origin_y + panel_height / 2,
                    y_label,
                )
            )

            for mode in combo:
                metrics = results[mode]
                color = COLOR_MAP[mode]
                mean_vals = metrics[metric_mean]
                points = []
                for idx, ratio_x in enumerate(round_positions):
                    x_pos = origin_x + ratio_x * panel_width
                    value = float(mean_vals[idx])
                    y_ratio = (value - min_val) / (max_val - min_val)
                    y_pos = origin_y + panel_height - y_ratio * panel_height
                    points.append(f"{x_pos:.2f},{y_pos:.2f}")

                svg_lines.append(
                    "  <polyline fill='none' stroke='{color}' stroke-width='3' points='{points_str}'/>".format(
                        color=color,
                        points_str=" ".join(points),
                    )
                )

                for idx, point in enumerate(points):
                    x_pos, y_pos = point.split(",")
                    svg_lines.append(
                        "  <circle cx='{x}' cy='{y}' r='5' fill='{color}'/>".format(
                            x=x_pos,
                            y=y_pos,
                            color=color,
                        )
                    )

        add_panel(
            origin_x=margin_x,
            origin_y=margin_y,
            bounds=accuracy_bounds,
            metric_mean="accuracy_mean",
            metric_std="accuracy_std",
            title="精度",
            y_label="Accuracy (%)",
        )

        add_panel(
            origin_x=margin_x + panel_width + panel_gap,
            origin_y=margin_y,
            bounds=time_bounds,
            metric_mean="round_time_mean",
            metric_std="round_time_std",
            title="実行時間",
            y_label="Round Time (seconds)",
        )

        legend_y = margin_y + panel_height + 80
        legend_x = margin_x
        svg_lines.append(
            "  <text x='{:.2f}' y='{:.2f}' class='legend' text-anchor='start'>凡例</text>".format(
                legend_x,
                legend_y,
            )
        )
        legend_y += 10
        for index, mode in enumerate(combo):
            color = COLOR_MAP[mode]
            label = {
                "plaintext": "Plaintext",
                "tfhe": "TFHE Encrypted",
                "ckks": "CKKS Encrypted",
            }[mode]
            entry_y = legend_y + (index + 1) * 30
            svg_lines.append(
                "  <rect x='{:.2f}' y='{:.2f}' width='24' height='24' fill='{color}'/>".format(
                    legend_x,
                    entry_y - 20,
                    color=color,
                )
            )
            svg_lines.append(
                "  <text x='{:.2f}' y='{:.2f}' text-anchor='start'>{}</text>".format(
                    legend_x + 36,
                    entry_y,
                    label,
                )
            )

        svg_lines.append("</svg>")

        filename = "benchmark_" + "_".join(combo) + ".svg"
        output_path = output_dir / filename
        output_path.write_text("\n".join(svg_lines), encoding="utf-8")
        saved_paths.append(output_path)

    return saved_paths


def plot_results(
    results: Dict[str, Dict[str, np.ndarray]],
    combos: Sequence[Sequence[str]],
    output_dir: Path,
    num_rounds: int,
) -> List[Path]:
    if plt is not None:
        round_axis = np.arange(1, num_rounds + 1)
        saved_paths: List[Path] = []

        for combo in combos:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("実験結果")

            for mode in combo:
                metrics = results[mode]
                label = {
                    "plaintext": "Plaintext",
                    "tfhe": "TFHE Encrypted",
                    "ckks": "CKKS Encrypted",
                }[mode]

                axes[0].plot(round_axis, metrics["accuracy_mean"], marker="o", label=label)
                axes[0].fill_between(
                    round_axis,
                    metrics["accuracy_mean"] - metrics["accuracy_std"],
                    metrics["accuracy_mean"] + metrics["accuracy_std"],
                    alpha=0.2,
                )

                axes[1].plot(round_axis, metrics["round_time_mean"], marker="o", label=label)
                axes[1].fill_between(
                    round_axis,
                    metrics["round_time_mean"] - metrics["round_time_std"],
                    metrics["round_time_mean"] + metrics["round_time_std"],
                    alpha=0.2,
                )

            axes[0].set_title("精度")
            axes[0].set_xlabel("Learning Round")
            axes[0].set_ylabel("Accuracy (%)")
            axes[0].grid(True, linestyle="--", alpha=0.5)

            axes[1].set_title("実行時間")
            axes[1].set_xlabel("Learning Round")
            axes[1].set_ylabel("Round Time (seconds)")
            axes[1].grid(True, linestyle="--", alpha=0.5)

            axes[0].legend()
            axes[1].legend()

            filename = "benchmark_" + "_".join(combo) + ".png"
            output_path = output_dir / filename
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(output_path)
            saved_paths.append(output_path)
            plt.close(fig)

        return saved_paths

    return _plot_results_svg(results, combos, output_dir, num_rounds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated aggregation benchmark")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per client")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of experiment repetitions")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=["plaintext", "tfhe", "ckks"],
        help="Aggregation modes to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_outputs"),
        help="Directory to save benchmark artefacts",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Optional JSON file to store raw metrics",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=2024,
        help="Base random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    available_modes = {"plaintext", "tfhe", "ckks"}
    modes = [mode.lower() for mode in args.modes if mode.lower() in available_modes]
    if not modes:
        raise ValueError("No valid aggregation modes were provided.")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in modes:
        print(f"Running benchmark for mode={mode}...")
        try:
            metrics = collect_metrics(
                aggregation_mode=mode,
                num_clients=args.clients,
                num_rounds=args.rounds,
                local_epochs=args.epochs,
                repetitions=args.repetitions,
                base_seed=args.base_seed,
            )
            results[mode] = metrics
        except ImportError as exc:
            print(f"Skipping mode '{mode}' because dependency is missing: {exc}")

    if not results:
        raise RuntimeError("No benchmarks were executed successfully.")

    combos: List[Tuple[str, ...]] = []
    if "plaintext" in results and "tfhe" in results:
        combos.append(("plaintext", "tfhe"))
    if "plaintext" in results and "ckks" in results:
        combos.append(("plaintext", "ckks"))
    if all(mode in results for mode in ("plaintext", "tfhe", "ckks")):
        combos.append(("plaintext", "tfhe", "ckks"))

    if combos:
        try:
            saved_paths = plot_results(results, combos, args.output, args.rounds)
            for path in saved_paths:
                print(f"Saved plot: {path}")
        except ImportError as exc:
            print(f"Skipping plot generation because matplotlib is unavailable: {exc}")
    else:
        print("No plot combinations available based on executed benchmarks.")

    if args.results_file:
        serialisable = {
            mode: {key: value.tolist() for key, value in metrics.items()}
            for mode, metrics in results.items()
        }
        with args.results_file.open("w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        print(f"Saved raw metrics to {args.results_file}")


if __name__ == "__main__":
    main()
