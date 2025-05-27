import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import matplotlib.pyplot as plt 

# Define a basic strategy (FedAvg)
# strategy = fl.server.strategy.FedAvg(
#     min_available_clients=2,  # Match the number of clients you plan to connect
# )


class CustomFedAvg(FedAvg):
    def __init__(self):
        super().__init__()
        self.accuracies = []
        self.losses = []

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[Round {server_round}] Received weights from clients:")
        
        # Show weights received from each client
        for i, (client, fit_res) in enumerate(results):
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            print(f"Client {i} weights:")
            for j, w in enumerate(client_weights):
                print(f"  Layer {j} weights:\n{w}")
                print(f"    mean={np.mean(w):.6f}, std={np.std(w):.6f}")


        # Perform the actual FedAvg aggregation
        aggregated_result = super().aggregate_fit(server_round, results, failures)

        if aggregated_result is not None:
            aggregated_weights = parameters_to_ndarrays(aggregated_result[0])
            print(f"\n[Round {server_round}] Aggregated (Global) Weights:")
            for i, w in enumerate(aggregated_weights):
                print(f"  Layer {i} weights:\n{w}")
                print(f"  Layer {i}: mean={np.mean(w):.6f}, std={np.std(w):.6f}")

        return aggregated_result
    
    def aggregate_evaluate(self, server_round, results, failures):
        print(f"\n[Round {server_round}] Evaluation Results:")
        
        accuracies = []
        losses = []
        num_examples = []

        for i, (client, eval_res) in enumerate(results):
            acc = eval_res.metrics.get("accuracy", None)
            loss = eval_res.loss
            n = eval_res.num_examples
            if acc is not None:
                print(f"  Client {i}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")
                accuracies.append(acc * n)
                losses.append(loss * n)
                num_examples.append(n)

        if accuracies and num_examples:
            weighted_acc = sum(accuracies) / sum(num_examples)
            weighted_loss = sum(losses) / sum(num_examples)
            self.accuracies.append(weighted_acc)
            self.losses.append(weighted_loss)
            print(f"  Global Accuracy (Weighted): {weighted_acc:.4f}")
            print(f"  Global Loss (Weighted): {weighted_loss:.4f}")

        return super().aggregate_evaluate(server_round, results, failures)

    
    def plot_metrics(self):
        rounds = list(range(1, len(self.accuracies) + 1))

        # Plot 1: Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, self.accuracies, marker='o', color='blue')
        plt.title("Global Accuracy Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.xticks(rounds)
        plt.savefig("global_accuracy_plot.png")
        plt.show()

        # Plot 2: Loss
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, self.losses, marker='x', color='green')
        plt.title("Global Loss Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.xticks(rounds)
        plt.savefig("global_loss_plot.png")
        plt.show()


# Start the server
strategy = CustomFedAvg()

# Start the server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    # strategy=CustomFedAvg(),
    strategy=strategy,
)

# Plot accuracy after training
strategy.plot_metrics()