"""Plaintext federated learning demonstration."""
from __future__ import annotations

from typing import List

from aggregators import PlaintextAggregator
from federated_core import Client, ClientConfig, FederatedServer, set_global_seed


def build_server(num_clients: int) -> FederatedServer:
    def factory(model, count):
        return PlaintextAggregator(model, count)

    return FederatedServer(num_clients=num_clients, aggregator_factory=factory, aggregator_name="Plaintext")


def main() -> None:
    print("=== Federated Learning (Plaintext Aggregation) ===")
    set_global_seed(0)

    num_clients = 3
    num_rounds = 5
    client_config = ClientConfig(learning_rate=0.1, local_epochs=2)

    server = build_server(num_clients)
    clients: List[Client] = [Client(client_id=i + 1, config=client_config, data_seed=i * 100) for i in range(num_clients)]

    print("\nInitial global model evaluation:")
    initial_accuracy, _ = server.evaluate_global_model()

    for round_idx in range(num_rounds):
        print(f"\n{'=' * 40}")
        print(f"ROUND {round_idx + 1}")
        print(f"{'=' * 40}")

        global_weights = server.get_global_weights()
        client_updates = []
        for client in clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            client_update = client.local_update(global_weights, verbose=True)
            client_updates.append(client_update)

        print("\n--- Server Aggregation ---")
        server.aggregate_models(client_updates)
        print(f"\nRound {round_idx + 1} Global Model Evaluation:")
        server.evaluate_global_model()

    print("\nFinal global model evaluation:")
    final_accuracy, _ = server.evaluate_global_model()
    print(f"Overall improvement: {final_accuracy - initial_accuracy:.2f}%")


if __name__ == "__main__":
    main()
