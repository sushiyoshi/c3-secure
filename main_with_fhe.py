"""Federated learning demonstration with TFHE encrypted aggregation."""
from __future__ import annotations

from typing import List

from aggregators import TFHEModelAggregator
from federated_core import Client, ClientConfig, FederatedServer, set_global_seed


def build_fhe_server(num_clients: int, verbose: bool = True) -> FederatedServer:
    def factory(model, count):
        return TFHEModelAggregator(model, count, scale_factor=100, max_value=50, verbose=verbose)

    return FederatedServer(
        num_clients=num_clients,
        aggregator_factory=factory,
        aggregator_name="TFHE",
    )


def main() -> None:
    print("=== 🔒 Federated Learning with TFHE Encryption ===")
    set_global_seed(0)

    num_clients = 5
    num_rounds = 10
    client_config = ClientConfig(learning_rate=0.01, local_epochs=2)

    server = build_fhe_server(num_clients=num_clients, verbose=True)
    clients: List[Client] = [Client(client_id=i + 1, config=client_config, data_seed=i * 100) for i in range(num_clients)]

    print("\nInitial global model weights:")
    for name, param in server.global_model.named_parameters():
        print(f"  {name}: {param.data}")

    print("\nInitial Global Model Evaluation:")
    initial_accuracy, _ = server.evaluate_global_model()

    for round_num in range(num_rounds):
        print(f"\n{'=' * 60}")
        print(f"🔒 FEDERATED LEARNING ROUND {round_num + 1} (TFHE ENCRYPTED)")
        print(f"{'=' * 60}")

        global_weights = server.get_global_weights()
        client_weights_list = []
        for client in clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            local_weights = client.local_update(global_weights, verbose=True)
            client_weights_list.append(local_weights)

        print(f"\n--- 🔒 FHE Server Aggregation ---")
        server.aggregate_models(client_weights_list)

        print(f"\nRound {round_num + 1} Global Model Evaluation:")
        server.evaluate_global_model()

        print(f"🔒 Encrypted Round {round_num + 1} completed!")

    print(f"\n{'=' * 60}")
    print("🔒 ENCRYPTED FEDERATED LEARNING COMPLETED!")
    print(f"{'=' * 60}")

    print("\nFinal global model weights:")
    for name, param in server.global_model.named_parameters():
        print(f"  {name}: {param.data}")

    print("\nFinal Global Model Evaluation:")
    final_accuracy, _ = server.evaluate_global_model()
    print(f"Overall improvement: {final_accuracy - initial_accuracy:.2f}%")


if __name__ == "__main__":
    main()
