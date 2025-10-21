"""Reusable federated learning components shared across scripts."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from aggregators import BaseModelAggregator, PlaintextAggregator


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic behaviour."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


class CustomDataset(Dataset):
    """Simple dataset backed by floating point tensors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def create_simple_data(
    batch_size: int,
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> DataLoader:
    """Generate a simple synthetic dataset for binary classification."""

    if seed is not None:
        np.random.seed(seed)

    features = np.random.randn(num_samples, 3).astype(np.float32)
    labels = ((features[:, 0] + features[:, 1] - features[:, 2]) > 0).astype(np.int64)
    dataset = CustomDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    """Single-layer neural network used for experiments."""

    def __init__(self, input_size: int = 3, output_size: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        with torch.no_grad():
            self.fc.weight.data = torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
            )
            self.fc.bias.data = torch.tensor([0.1, 0.2], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - tiny wrapper
        return self.fc(x)


class LocalTrainer:
    """Train and evaluate a client model using local data."""

    def __init__(
        self,
        model: nn.Module,
        dataset: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, epochs: int, verbose: bool = True) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for data, target in self.dataset:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                predicted = torch.argmax(output, dim=1)
                epoch_correct += (predicted == target).sum().item()
                epoch_samples += target.size(0)
                epoch_loss += loss.item()

            epoch_accuracy = epoch_correct / max(epoch_samples, 1) * 100
            if verbose:
                print(
                    f"    Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%"
                )

            total_loss += epoch_loss
            correct += epoch_correct
            total_samples += epoch_samples

        overall_accuracy = correct / max(total_samples, 1) * 100
        if verbose:
            print(f"  Local training completed: Overall Accuracy = {overall_accuracy:.2f}%")
        return overall_accuracy

    def evaluate(self, dataset: Optional[DataLoader] = None, verbose: bool = True) -> tuple[float, float]:
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        target_dataset = dataset if dataset is not None else self.dataset

        with torch.no_grad():
            for data, target in target_dataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()

        accuracy = correct / max(total, 1) * 100
        avg_loss = total_loss / max(len(target_dataset), 1)
        if verbose:
            print(f"  Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def set_model_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(weights)


@dataclass
class ClientConfig:
    learning_rate: float = 0.01
    batch_size: int = 16
    train_samples: int = 100
    test_samples: int = 50
    local_epochs: int = 2


class Client:
    """Federated learning client participating in rounds."""

    def __init__(self, client_id: int, config: ClientConfig, data_seed: Optional[int] = None) -> None:
        self.client_id = client_id
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NeuralNetwork(input_size=3, output_size=2).to(self.device)
        self.train_loader = create_simple_data(
            batch_size=config.batch_size,
            num_samples=config.train_samples,
            seed=data_seed,
        )
        self.test_loader = create_simple_data(
            batch_size=config.batch_size,
            num_samples=config.test_samples,
            seed=None if data_seed is None else data_seed + 1000,
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.trainer = LocalTrainer(self.model, self.train_loader, self.optimizer, self.criterion, self.device)

    def local_update(
        self,
        global_weights: Dict[str, torch.Tensor],
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if verbose:
            print(f"Client {self.client_id}: Starting local update")

        self.trainer.set_model_weights(global_weights)

        if verbose:
            print(f"Client {self.client_id}: Pre-training evaluation:")
            pre_accuracy, _ = self.trainer.evaluate(self.test_loader, verbose=verbose)
            print(f"Client {self.client_id}: Starting local training...")
        else:
            pre_accuracy = None

        train_accuracy = self.trainer.train(self.config.local_epochs, verbose=verbose)

        if verbose:
            print(f"Client {self.client_id}: Post-training evaluation:")
            post_accuracy, _ = self.trainer.evaluate(self.test_loader, verbose=verbose)
            print(f"Client {self.client_id} Summary:")
            print(f"  Pre-training accuracy: {pre_accuracy:.2f}%")
            print(f"  Training accuracy: {train_accuracy:.2f}%")
            print(f"  Post-training accuracy: {post_accuracy:.2f}%")
            print(f"  Accuracy improvement: {post_accuracy - pre_accuracy:.2f}%")

        return self.trainer.get_model_weights()

    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        return self.trainer.get_model_weights()


class FederatedServer:
    """Server coordinating federated learning rounds."""

    def __init__(
        self,
        num_clients: int,
        aggregator_factory: Callable[[torch.nn.Module, int], BaseModelAggregator],
        aggregator_name: str,
        test_seed: int = 9999,
    ) -> None:
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = NeuralNetwork(input_size=3, output_size=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.test_loader = create_simple_data(batch_size=32, num_samples=200, seed=test_seed)
        self.aggregator_name = aggregator_name
        self.aggregator = aggregator_factory(self.global_model, num_clients)

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.global_model.state_dict())

    def set_global_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        self.global_model.load_state_dict(weights)

    def aggregate_models(self, client_weights_list: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        aggregated_state = self.aggregator.aggregate(client_weights_list)
        self.global_model.load_state_dict(aggregated_state)
        return aggregated_state

    def evaluate_global_model(self, verbose: bool = True) -> tuple[float, float]:
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = self.criterion(output, target)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()

        accuracy = correct / max(total, 1) * 100
        avg_loss = total_loss / max(len(self.test_loader), 1)
        if verbose:
            print(f"Global Model ({self.aggregator_name}) - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss


def build_plaintext_server(num_clients: int, test_seed: int = 9999) -> FederatedServer:
    """Convenience constructor for plaintext aggregation."""

    def factory(model: torch.nn.Module, count: int) -> BaseModelAggregator:
        return PlaintextAggregator(model, count)

    return FederatedServer(num_clients=num_clients, aggregator_factory=factory, aggregator_name="Plaintext", test_seed=test_seed)


__all__ = [
    "Client",
    "ClientConfig",
    "FederatedServer",
    "NeuralNetwork",
    "LocalTrainer",
    "PlaintextAggregator",
    "build_plaintext_server",
    "create_simple_data",
    "set_global_seed",
]
