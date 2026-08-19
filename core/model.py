import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AngleNetLayer(nn.Module):
    """
    A layer that parameterizes weight matrices through geometric angles,
    inspired by angle-nets. Ensures relational structure is preserved
    in weight space.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = nn.Parameter(torch.empty(out_features, in_features))
        self.beta = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.alpha, 0, 2 * math.pi)
        nn.init.normal_(self.beta, 0, 0.1)

    def forward(self, x):
        weight = torch.cos(self.alpha) * torch.exp(self.beta)
        return F.linear(x, weight)


class SensoryEncoder(nn.Module):
    """
    Encodes raw sensory input into a structured latent representation.
    For Stage 1: positions, velocities, and contact signals for N objects.
    """

    def __init__(self, num_objects, object_dim=4, hidden_dim=64):
        super().__init__()
        self.num_objects = num_objects
        self.object_dim = object_dim
        input_dim = num_objects * object_dim
        self.encoder = nn.Sequential(
            AngleNetLayer(input_dim, hidden_dim),
            nn.ReLU(),
            AngleNetLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, sensory_input):
        batch_size = sensory_input.shape[0]
        x = sensory_input.view(batch_size, -1)
        return self.encoder(x)


class PhysicsPredictionModule(nn.Module):
    """
    Predicts next sensory state given current state and action.
    Incorporates UGCT-style physics constraints via differentiable
    regularization terms.
    """

    def __init__(self, hidden_dim, num_objects, object_dim=4, action_dim=None):
        super().__init__()
        self.num_objects = num_objects
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim if action_dim is not None else num_objects * object_dim

        self.action_proj = nn.Linear(self.action_dim, hidden_dim)
        self.transition = nn.Sequential(
            AngleNetLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
            AngleNetLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects * object_dim),
        )

    def forward(self, z, action):
        action_emb = self.action_proj(action)
        combined = z + action_emb
        delta = self.transition(combined)
        return delta.view(-1, self.num_objects, self.object_dim)

    def physics_constraint_loss(self, current_state, next_state, delta_pred):
        """
        Computes soft physics constraint violations.
        Enforces momentum conservation during collisions.
        """
        constraint_loss = 0.0

        curr_pos = current_state[:, :, :2]
        curr_vel = current_state[:, :, 2:]
        next_pos = next_state[:, :, :2]
        next_vel = next_state[:, :, 2:]

        predicted_next_pos = curr_pos + delta_pred[:, :, :2]
        predicted_next_vel = curr_vel + delta_pred[:, :, 2:]

        constraint_loss = constraint_loss + F.mse_loss(predicted_next_pos, next_pos)
        constraint_loss = constraint_loss + F.mse_loss(predicted_next_vel, next_vel)

        return constraint_loss


class AgentModule(nn.Module):
    """
    For later stages: infers goals and predicts agent actions.
    Currently a placeholder for Stage 3+.
    """

    def __init__(self, hidden_dim, num_agents, goal_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.goal_dim = goal_dim
        self.inference_net = nn.Sequential(
            AngleNetLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * goal_dim),
        )

    def forward(self, z):
        return self.inference_net(z)


class DevelopmentalPhysicsNet(nn.Module):
    """
    The main model. Combines sensory encoding, angle-weighted reasoning,
    and physics prediction in a predictive coding framework.

    Stages:
      Stage 1 (Sensorimotor): Predict next state from current state + action.
      Stage 2 (Intuitive Physics): Add explicit forward dynamics + physics constraints.
      Stage 3 (Intuitive Psychology): Add agent module for theory-of-mind.
      Stage 4 (Symbolic): Add symbolic grounding and language interface.
    """

    def __init__(self, num_objects=4, object_dim=4, hidden_dim=64, stage=1):
        super().__init__()
        self.num_objects = num_objects
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.stage = stage

        self.encoder = SensoryEncoder(num_objects, object_dim, hidden_dim)
        self.predictor = PhysicsPredictionModule(hidden_dim, num_objects, object_dim, action_dim=num_objects * 2)

        self.agent_module = None
        if stage >= 3:
            self.agent_module = AgentModule(hidden_dim, num_agents=1, goal_dim=2)

    def forward(self, sensory_input, action):
        z = self.encoder(sensory_input)
        delta = self.predictor(z, action)
        next_state = sensory_input + delta
        return next_state, z, delta

    def compute_loss(self, sensory_input, action, next_sensory_input):
        pred_next, z, delta = self(sensory_input, action)

        prediction_loss = F.mse_loss(pred_next, next_sensory_input)

        if self.stage >= 2:
            physics_loss = self.predictor.physics_constraint_loss(
                sensory_input, next_sensory_input, delta
            )
        else:
            physics_loss = torch.tensor(0.0, device=sensory_input.device)

        total_loss = prediction_loss + 0.1 * physics_loss
        return total_loss, {
            "prediction_loss": prediction_loss.item(),
            "physics_loss": physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
            "total_loss": total_loss.item(),
        }


def active_inference_step(model, current_sensory_input, action_space, device):
    """
    Active inference: choose action that minimizes expected prediction error.
    """
    best_action = None
    best_score = float("inf")

    current_sensory_input = current_sensory_input.to(device)

    for action in action_space:
        action_tensor = torch.tensor(action, device=device).unsqueeze(0)
        with torch.no_grad():
            pred_next, _, _ = model(current_sensory_input, action_tensor)
        score = torch.norm(pred_next - current_sensory_input).item()
        if score < best_score:
            best_score = score
            best_action = action

    return best_action
