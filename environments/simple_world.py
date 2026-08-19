import numpy as np


class SimpleWorld:
    """
    A deterministic 2D physics world with balls, gravity, and collisions.
    Provides sensory input (positions, velocities, contact forces) for training
    the developmental model.

    State representation:
      For each of N balls: [x, y, vx, vy]
      Plus a contact flag per pair (N*N matrix, upper triangular)

    Dynamics:
      - Gravity: vy += g * dt
      - Position: x += vx * dt, y += vy * dt
      - Floor/wall collisions: reflect velocity with restitution
      - Ball-ball collisions: elastic collision resolution
    """

    def __init__(
        self,
        num_balls=3,
        dt=0.02,
        gravity=9.81,
        restitution=0.8,
        world_size=(10.0, 10.0),
        seed=42,
    ):
        self.num_balls = num_balls
        self.dt = dt
        self.gravity = gravity
        self.restitution = restitution
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        """Initialize balls with random positions and velocities."""
        positions = self.rng.uniform(
            low=[1.0, 1.0], high=[self.world_size[0] - 1.0, self.world_size[1] - 1.0],
            size=(self.num_balls, 2)
        )
        velocities = self.rng.uniform(
            low=[-2.0, -2.0], high=[2.0, 2.0],
            size=(self.num_balls, 2)
        )
        self.state = np.concatenate([positions, velocities], axis=1).astype(np.float32)
        self.contact_forces = np.zeros((self.num_balls, self.num_balls), dtype=np.float32)
        return self.get_sensory_input()

    def get_sensory_input(self):
        """
        Returns:
          state: (num_balls, 4) array of [x, y, vx, vy]
          contacts: (num_balls, num_balls) contact force matrix (0 if no contact)
        """
        return self.state.copy(), self.contact_forces.copy()

    def step(self, action=None):
        """
        Advance physics by one timestep.

        Args:
          action: optional array of shape (num_balls * 2,) for forces applied
                  to each ball [fx1, fy1, fx2, fy2, ...]

        Returns:
          sensory_input: (state, contacts)
        """
        if action is None:
            action = np.zeros(self.num_balls * 2, dtype=np.float32)

        forces = np.zeros((self.num_balls, 2), dtype=np.float32)
        forces[:, 1] -= self.gravity

        if action is not None:
            action = np.asarray(action, dtype=np.float32).reshape(self.num_balls, 2)
            forces += action

        self.contact_forces[:] = 0.0

        for i in range(self.num_balls):
            self.state[i, 2:] += forces[i] * self.dt
            self.state[i, :2] += self.state[i, 2:] * self.dt

        for i in range(self.num_balls):
            for axis in range(2):
                if self.state[i, axis] < 0.25:
                    self.state[i, axis] = 0.25
                    self.state[i, 2 + axis] *= -self.restitution
                elif self.state[i, axis] > self.world_size[axis] - 0.25:
                    self.state[i, axis] = self.world_size[axis] - 0.25
                    self.state[i, 2 + axis] *= -self.restitution

        for i in range(self.num_balls):
            for j in range(i + 1, self.num_balls):
                dx = self.state[j, 0] - self.state[i, 0]
                dy = self.state[j, 1] - self.state[i, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = 0.5

                if dist < min_dist and dist > 1e-6:
                    nx = dx / dist
                    ny = dy / dist
                    overlap = min_dist - dist

                    self.state[i, 0] -= nx * overlap / 2
                    self.state[i, 1] -= ny * overlap / 2
                    self.state[j, 0] += nx * overlap / 2
                    self.state[j, 1] += ny * overlap / 2

                    rel_vel_x = self.state[i, 2] - self.state[j, 2]
                    rel_vel_y = self.state[i, 3] - self.state[j, 3]
                    rel_vel_normal = rel_vel_x * nx + rel_vel_y * ny

                    if rel_vel_normal > 0:
                        impulse = rel_vel_normal * (1 + self.restitution) / 2
                        self.state[i, 2] -= impulse * nx
                        self.state[i, 3] -= impulse * ny
                        self.state[j, 2] += impulse * nx
                        self.state[j, 3] += impulse * ny
                        self.contact_forces[i, j] = impulse
                        self.contact_forces[j, i] = impulse

        return self.get_sensory_input()

    def render(self):
        return self.state.copy()
