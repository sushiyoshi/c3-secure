"""Benchmark plaintext vs FHE-based federated aggregation strategies."""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt

from aggregators import CKKSModelAggregator, PlaintextAggregator, TFHEModelAggregator
from federated_core import Client, ClientConfig, FederatedServer, set_global_seed


@dataclass
class RoundMetrics:
    round_number: int
    accuracy: float
    runtime_seconds: float


@dataclass
class AggregatorResult:
    name: str
    rounds: List[RoundMetrics]
    initial_accuracy: float
    final_accuracy: float

    @property
    def average_runtime(self) -> float:
        return statistics.mean(metric.runtime_seconds for metric in self.rounds)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "rounds": [metric.__dict__ for metric in self.rounds],
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "average_runtime": self.average_runtime,
        }


AggregatorFactory = Callable[[int], FederatedServer]


def _make_plaintext_server(num_clients: int) -> FederatedServer:
    def factory(model, count):
        return PlaintextAggregator(model, count)

    return FederatedServer(num_clients=num_clients, aggregator_factory=factory, aggregator_name="Plaintext")


def _make_tfhe_server(num_clients: int) -> FederatedServer:
    def factory(model, count):
        return TFHEModelAggregator(model, count, scale_factor=100, max_value=50, verbose=False)

    return FederatedServer(num_clients=num_clients, aggregator_factory=factory, aggregator_name="TFHE")


def _make_ckks_server(num_clients: int) -> FederatedServer:
    def factory(model, count):
        return CKKSModelAggregator(model, count, scale_factor=1.0, verbose=False)

    return FederatedServer(num_clients=num_clients, aggregator_factory=factory, aggregator_name="CKKS")


AGGREGATOR_BUILDERS: Dict[str, AggregatorFactory] = {
    "plaintext": _make_plaintext_server,
    "tfhe": _make_tfhe_server,
    "ckks": _make_ckks_server,
}


def run_experiment(
    name: str,
    builder: AggregatorFactory,
    num_clients: int,
    num_rounds: int,
    client_config: ClientConfig,
    seed: int,
    verbose: bool = False,
) -> AggregatorResult:
    set_global_seed(seed)
    server = builder(num_clients)
    clients: List[Client] = [Client(client_id=i + 1, config=client_config, data_seed=i * 100) for i in range(num_clients)]

    initial_accuracy, _ = server.evaluate_global_model(verbose=verbose)
    round_metrics: List[RoundMetrics] = []

    for round_index in range(1, num_rounds + 1):
        round_start = time.perf_counter()
        global_weights = server.get_global_weights()
        client_updates = [client.local_update(global_weights, verbose=verbose) for client in clients]
        server.aggregate_models(client_updates)
        accuracy, _ = server.evaluate_global_model(verbose=verbose)
        round_elapsed = time.perf_counter() - round_start
        round_metrics.append(RoundMetrics(round_number=round_index, accuracy=accuracy, runtime_seconds=round_elapsed))

    final_accuracy = round_metrics[-1].accuracy if round_metrics else initial_accuracy
    return AggregatorResult(name=name, rounds=round_metrics, initial_accuracy=initial_accuracy, final_accuracy=final_accuracy)


def validate_plaintext_superiority(results: Mapping[str, AggregatorResult], tolerance: float = 1e-3) -> List[Dict[str, str]]:
    validations: List[Dict[str, str]] = []
    plaintext = results.get("plaintext")
    if plaintext is None:
        return validations

    for name, result in results.items():
        if name == "plaintext":
            continue

        status = "PASS"
        messages: List[str] = []

        if result.final_accuracy > plaintext.final_accuracy + tolerance:
            status = "FAIL"
            messages.append(
                "Encrypted final accuracy exceeded plaintext final accuracy; investigate potential inconsistency."
            )

        if result.average_runtime < plaintext.average_runtime - tolerance:
            status = "FAIL"
            messages.append(
                "Encrypted average runtime was shorter than plaintext runtime; verify benchmarking procedure."
            )

        validations.append({"comparator": name, "status": status, "details": " ".join(messages) or "OK"})

    return validations


def write_log(
    output_path: Path,
    combination_name: str,
    results: Mapping[str, AggregatorResult],
    validations: Iterable[Dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Federated Learning Benchmark: {combination_name}\n")
        handle.write("=" * 72 + "\n\n")
        for name, result in results.items():
            handle.write(f"Aggregator: {name}\n")
            handle.write(f"  Initial Accuracy: {result.initial_accuracy:.4f}%\n")
            handle.write(f"  Final Accuracy:   {result.final_accuracy:.4f}%\n")
            handle.write(f"  Average Runtime:  {result.average_runtime:.4f} seconds\n")
            handle.write("  Round Metrics:\n")
            for metric in result.rounds:
                handle.write(
                    f"    Round {metric.round_number:02d}: Accuracy={metric.accuracy:.4f}% | Runtime={metric.runtime_seconds:.4f}s\n"
                )
            handle.write("\n")

        handle.write("Validation Summary\n")
        handle.write("-" * 72 + "\n")
        for validation in validations:
            handle.write(
                f"Comparator: {validation['comparator']:<8} Status: {validation['status']:<4} Details: {validation['details']}\n"
            )
        handle.write("\n")

        structured_payload = {name: result.to_dict() for name, result in results.items()}
        handle.write("RESULTS_JSON: " + json.dumps(structured_payload) + "\n")


def parse_structured_results(log_path: Path) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("RESULTS_JSON: "):
                payload = line.split("RESULTS_JSON: ", 1)[1]
                data = json.loads(payload)
                break
    return data


def generate_plots(
    output_dir: Path,
    combination_name: str,
    results: Mapping[str, AggregatorResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = [metric.round_number for metric in next(iter(results.values())).rounds]

    # Runtime plot
    plt.figure(figsize=(8, 4.5))
    for name, result in results.items():
        times = [metric.runtime_seconds for metric in result.rounds]
        plt.plot(rounds, times, marker="o", label=name.capitalize())
    plt.title("実験結果\n実行時間")
    plt.xlabel("Learning Round")
    plt.ylabel("Round Time (seconds)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    runtime_path = output_dir / f"{combination_name}_runtime.png"
    plt.tight_layout()
    plt.savefig(runtime_path)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 4.5))
    for name, result in results.items():
        accuracies = [metric.accuracy for metric in result.rounds]
        plt.plot(rounds, accuracies, marker="s", label=name.capitalize())
    plt.title("実験結果\n精度")
    plt.xlabel("Learning Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    accuracy_path = output_dir / f"{combination_name}_accuracy.png"
    plt.tight_layout()
    plt.savefig(accuracy_path)
    plt.close()


def run_combination(
    combination_name: str,
    aggregator_keys: List[str],
    num_clients: int,
    num_rounds: int,
    client_config: ClientConfig,
    seed: int,
    output_dir: Path,
) -> Path:
    results: Dict[str, AggregatorResult] = {}

    for key in aggregator_keys:
        builder = AGGREGATOR_BUILDERS[key]
        try:
            result = run_experiment(
                name=key,
                builder=builder,
                num_clients=num_clients,
                num_rounds=num_rounds,
                client_config=client_config,
                seed=seed,
                verbose=False,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"Aggregator '{key}' is unavailable. Ensure all dependencies are installed."
            ) from exc

        results[key] = result

    validations = validate_plaintext_superiority(results)
    log_path = output_dir / f"{combination_name}.txt"
    write_log(log_path, combination_name, results, validations)
    generate_plots(output_dir, combination_name, results)

    # Parse the log back to verify structured payload integrity
    _ = parse_structured_results(log_path)
    return log_path


def main() -> None:
    output_dir = Path("benchmark_results")
    num_clients = 5
    num_rounds = 10
    client_config = ClientConfig(learning_rate=0.01, local_epochs=2)
    seed = 42

    combinations = {
        "plaintext_vs_tfhe": ["plaintext", "tfhe"],
        "plaintext_vs_ckks": ["plaintext", "ckks"],
        "plaintext_vs_tfhe_vs_ckks": ["plaintext", "tfhe", "ckks"],
    }

    for combo_name, keys in combinations.items():
        print(f"Running benchmark combination: {combo_name}")
        try:
            log_path = run_combination(
                combination_name=combo_name,
                aggregator_keys=keys,
                num_clients=num_clients,
                num_rounds=num_rounds,
                client_config=client_config,
                seed=seed,
                output_dir=output_dir,
            )
            print(f"  Results written to {log_path}")
        except RuntimeError as exc:
            print(f"  ⚠️ Skipping combination '{combo_name}': {exc}")


if __name__ == "__main__":
    main()
