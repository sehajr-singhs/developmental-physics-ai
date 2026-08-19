import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import DevelopmentalPhysicsNet
from environments.simple_world import SimpleWorld


def train_stage1(
    num_epochs=200,
    steps_per_epoch=50,
    num_balls=3,
    lr=1e-3,
    device=None,
    save_path="experiments/results/stage1_results.json",
):
    """
    Stage 1: Sensorimotor Learning
    Trains the model to predict object positions and collisions in the simple 2D world.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world = SimpleWorld(num_balls=num_balls, seed=42)
    model = DevelopmentalPhysicsNet(
        num_objects=num_balls, object_dim=4, hidden_dim=64, stage=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "total_loss": [], "prediction_loss": [], "physics_loss": []}

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        world.reset()
        epoch_losses = {"total_loss": [], "prediction_loss": [], "physics_loss": []}

        for _ in range(steps_per_epoch):
            current_sensory, _ = world.get_sensory_input()
            current_tensor = torch.tensor(current_sensory, device=device).unsqueeze(0)

            action = world.rng.uniform(-1.0, 1.0, size=(num_balls * 2,)).astype(np.float32)
            action_tensor = torch.tensor(action, device=device).unsqueeze(0)

            next_sensory, _ = world.step(action)
            next_tensor = torch.tensor(next_sensory, device=device).unsqueeze(0)

            optimizer.zero_grad()
            total_loss, metrics = model.compute_loss(current_tensor, action_tensor, next_tensor)
            total_loss.backward()
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key].append(metrics[key])

        avg_total = np.mean(epoch_losses["total_loss"])
        avg_pred = np.mean(epoch_losses["prediction_loss"])
        avg_phys = np.mean(epoch_losses["physics_loss"])

        history["epoch"].append(epoch + 1)
        history["total_loss"].append(float(avg_total))
        history["prediction_loss"].append(float(avg_pred))
        history["physics_loss"].append(float(avg_phys))

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1:4d}/{num_epochs} | "
                f"Total: {avg_total:.6f} | "
                f"Pred: {avg_pred:.6f} | "
                f"Phys: {avg_phys:.6f}"
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {save_path}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["epoch"], history["total_loss"], label="Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["epoch"], history["prediction_loss"], label="Prediction", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Prediction Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["epoch"], history["physics_loss"], label="Physics", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Physics Loss")
    plt.legend()

    plt.tight_layout()
    plot_path = save_path.replace(".json", ".png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()

    return model, history


if __name__ == "__main__":
    model, history = train_stage1(num_epochs=200, steps_per_epoch=50, num_balls=3)
    print("\nStage 1 training complete.")
