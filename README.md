# N-Body Simulator (CUDA)

An N-Body simulator powered by CUDA devices. This simulator models the gravitational interactions between particles in a 3D space with various initial configurations. It is currently under active development.

## Status
🚧 **Work in Progress**  
This project is still in its early stages. Some features have been implemented, while others are still being developed. The current features include:

- Gravitational force calculation using pairwise interactions.
- Velocity kernel with dynamic velocity adjustment based on particle distance.
- Minimum distance threshold for stable simulations.
- Box-Muller particle initialization for random distribution.

![Simulation Animation](docs/sim_anim_threshold_gravity_projections_gif2.gif)

These visualizations were rendered using [Makie.jl](https://docs.makie.org/v0.21/) and are not part of this project (although they may be included in the future idk).

![Simulation Animation](postprocess/sim_anim_threshold_gravity.gif)

## Future Work
- Implement force optimizations
- Improve stability in high-density scenarios
- Add post-processing tools for result visualization
- Collision detection and Post-collision corrections

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Installation and Usage
Once the project reaches a more stable state.
