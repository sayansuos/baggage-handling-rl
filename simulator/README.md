# Baggage Handling with RL Optimization

## Simulator Status: 21/05/2026

**Implemented:**

- [x] Random environment generation: static obstacles, moving obstacles, multiple agents, random start and target positions.

- [x] Basic motion system: agents and moving obstacles follow A* paths computed at scenario initialization until the end of the episode.

- [x] Observation system: local grid-based observations, relative distances and positions, robot state (velocity, orientation, etc.).

- [x] Reward function: includes goal progress, collision penalties, safety distances, and agent rotations following the experimental plan.

**In progress:**

- [ ] Fix reward function
- [x] Add collision detection logic

**TODO before SAC integration:**

- [x] Define termination conditions
- [x] Compute average time travel
- [ ] Validate observation space, action space, and reward function.
- [ ] Set pre-defined environment configurations
- [x] Add logs