# Developmental Physics AI

An AI that understands physics and the universe the way a child learns — through raw sensory input, interaction, and the gradual construction of intuitive theories.

## Core Hypothesis

Current AI systems lack the **core knowledge** that human infants possess: an intuitive understanding of objects, agents, forces, space, and time. We propose building AI by reverse-engineering how children learn physics — not from labeled datasets, but from unstructured interaction with the world.

## The Four Stages

1. **Sensorimotor** — Learn object permanence, spatial relations, and basic dynamics through vision and touch.
2. **Intuitive Physics** — Learn force, gravity, collision, and support by actively manipulating objects.
3. **Intuitive Psychology** — Learn agents, goals, beliefs, and intentions through social interaction.
4. **Symbolic Reasoning** — Learn language, mathematics, and abstract rules grounded in physical experience.

## Technical Approach

We combine:
- **Angle Geometry** (from angle-nets): geometric reasoning via structured weight spaces
- **Physics Constraints** (from UGCT): differentiable physical priors and symmetries
- **Predictive Coding**: minimize prediction error across sensory hierarchies
- **Active Inference**: act to reduce uncertainty and test hypotheses

## Project Structure

```
core/            — Model architectures and training logic
environments/    — Simulated worlds for embodied interaction
experiments/     — Training and evaluation scripts
paper/           — Manuscripts and theoretical work
notebooks/       — Exploratory analysis
```

## Quick Start

```bash
pip install -r requirements.txt
python experiments/stage1_sensorimotor.py
```

## Vision

We believe that true machine understanding — of physics, of mind, of language — cannot emerge from passive pattern matching alone. It requires a body, a world to act in, and time to build theories from the ground up.
