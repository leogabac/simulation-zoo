import argparse
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import fubumio as fm
from sysentropy import LoggerConfig, get_logger

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

# sentinel object
_UNSET = object()

PARTICLE_PALETTE = (
    fm.palettes.paper[0],
    fm.palettes.paper[1],
    fm.palettes.paper[2],
    fm.palettes.paper[3],
    fm.palettes.paper[4],
)

TYPE_MAP = {
    type_index: color
    for type_index, color in enumerate(PARTICLE_PALETTE)
}


def _identity_decorator(function):
    return function


_njit = njit if njit is not None else _identity_decorator


@_njit
def _compute_particle_life_forces(
    coords: np.ndarray,
    types: np.ndarray,
    interaction_matrix: np.ndarray,
    box_length: float,
    interaction_radius: float,
    repulsion_radius: float,
    force_scale: float,
) -> np.ndarray:
    n_particles = coords.shape[0]
    forces = np.zeros((n_particles, 2), dtype=np.float64)

    for i in range(n_particles):
        for j in range(n_particles):
            if i == j:
                continue

            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]

            dx = dx - box_length * np.round(dx / box_length)
            dy = dy - box_length * np.round(dy / box_length)

            distance = np.sqrt(dx * dx + dy * dy)
            if distance == 0.0 or distance >= interaction_radius:
                continue

            if distance < repulsion_radius:
                scale = force_scale * (distance / repulsion_radius - 1.0) / distance
            else:
                radial_factor = 1.0 - (
                    (distance - repulsion_radius)
                    / (interaction_radius - repulsion_radius)
                )
                strength = interaction_matrix[types[i], types[j]]
                scale = force_scale * strength * radial_factor / distance
            forces[i, 0] += scale * dx
            forces[i, 1] += scale * dy

    return forces


@_njit
def _compute_particle_life_forces_cell_list(
    coords: np.ndarray,
    types: np.ndarray,
    interaction_matrix: np.ndarray,
    box_length: float,
    interaction_radius: float,
    repulsion_radius: float,
    force_scale: float,
) -> np.ndarray:
    n_particles = coords.shape[0]
    forces = np.zeros((n_particles, 2), dtype=np.float64)

    n_cells = max(1, int(np.floor(box_length / interaction_radius)))
    cell_size = box_length / n_cells
    head = np.full(n_cells * n_cells, -1, dtype=np.int64)
    next_index = np.full(n_particles, -1, dtype=np.int64)
    cell_x = np.empty(n_particles, dtype=np.int64)
    cell_y = np.empty(n_particles, dtype=np.int64)

    for i in range(n_particles):
        cx = int(np.floor(coords[i, 0] / cell_size)) % n_cells
        cy = int(np.floor(coords[i, 1] / cell_size)) % n_cells
        cell_x[i] = cx
        cell_y[i] = cy
        cell_id = cy * n_cells + cx
        next_index[i] = head[cell_id]
        head[cell_id] = i

    for i in range(n_particles):
        base_cx = cell_x[i]
        base_cy = cell_y[i]

        for offset_y in (-1, 0, 1):
            neighbor_cy = (base_cy + offset_y) % n_cells
            for offset_x in (-1, 0, 1):
                neighbor_cx = (base_cx + offset_x) % n_cells
                cell_id = neighbor_cy * n_cells + neighbor_cx
                j = head[cell_id]

                while j != -1:
                    if i != j:
                        dx = coords[j, 0] - coords[i, 0]
                        dy = coords[j, 1] - coords[i, 1]

                        dx = dx - box_length * np.round(dx / box_length)
                        dy = dy - box_length * np.round(dy / box_length)

                        distance = np.sqrt(dx * dx + dy * dy)
                        if distance != 0.0 and distance < interaction_radius:
                            if distance < repulsion_radius:
                                scale = (
                                    force_scale
                                    * (distance / repulsion_radius - 1.0)
                                    / distance
                                )
                            else:
                                radial_factor = 1.0 - (
                                    (distance - repulsion_radius)
                                    / (interaction_radius - repulsion_radius)
                                )
                                strength = interaction_matrix[types[i], types[j]]
                                scale = (
                                    force_scale * strength * radial_factor / distance
                                )
                            forces[i, 0] += scale * dx
                            forces[i, 1] += scale * dy

                    j = next_index[j]

    return forces


@_njit
def _advance_particle_life_euler_inplace(
    coords: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    types: np.ndarray,
    interaction_matrix: np.ndarray,
    box_length: float,
    interaction_radius: float,
    repulsion_radius: float,
    force_scale: float,
    damping_coefficient: float,
    dt: float,
    n_substeps: int,
) -> None:
    for _ in range(n_substeps):
        forces = _compute_particle_life_forces_cell_list(
            coords,
            types,
            interaction_matrix,
            box_length,
            interaction_radius,
            repulsion_radius,
            force_scale,
        )
        accelerations = forces / masses[:, None]
        velocities += dt * (accelerations - damping_coefficient * velocities)
        coords += dt * velocities
        coords %= box_length


@dataclass
class PeriodicSquareBox2D:
    """
    2d square simulation box with periodic boundary conditions.

    the wrapped coordinate is
    x -> x mod L,
    and the minimum-image displacement is
    Δr -> Δr - L*round(Δr / L)`.
    """

    box_length: float

    def __post_init__(self) -> None:
        self.half_box_length = self.box_length / 2

    def wrap(self, coords: np.ndarray) -> np.ndarray:
        """
        wrap coordinates into [0, L).
        """
        wrapped = np.mod(coords, self.box_length)
        return np.where(np.isclose(wrapped, self.box_length), 0.0, wrapped)

    def displacement(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        minimum-image displacement vector r2 - r1.

        Δr -> Δr - L*round(Δr / L)`.
        """

        dr = np.asarray(coords2) - np.asarray(coords1)
        dr = dr - self.box_length * np.round(dr / self.box_length)
        return dr

    def distance(
        self, coords1: np.ndarray, coords2: np.ndarray, with_disp: bool = False
    ) -> float | tuple[float, np.ndarray]:
        """
        minimum-image distance.
        """
        dr = self.displacement(coords1, coords2)
        distance = np.linalg.norm(dr, axis=-1)
        if with_disp:
            return distance, dr

        return distance

    @property
    def area(self) -> float:
        return self.box_length**2

    @property
    def volume(self) -> float:
        return self.area

    def random_coordinate(self):
        return np.random.uniform(0, self.box_length, size=(2,))


class Particle:
    """
    point particle carrying mass, position, and possibly velocity.

    the stored state is `(m, t, x, v)`, where later verlet states may leave
    `v` unset even though the initial slice still needs `v_0`.
    """

    def __init__(
        self,
        mass: float,
        time: float,
        coords: list[float] | np.ndarray,
        vel: list[float] | np.ndarray | object,
        type_: int,
    ) -> None:

        self.time = time
        self.mass = mass
        self.coords = np.asarray(coords, dtype=float)
        self.vel = _UNSET if vel is _UNSET else np.asarray(vel, dtype=float)
        self.type = type_
        self.color = TYPE_MAP[type_]

    def x(self) -> float:
        return self.coords[0]

    def y(self) -> float:
        return self.coords[1]

    def vx(self) -> float | object:
        if self.vel is _UNSET:
            return _UNSET
        return self.vel[0]

    def vy(self) -> float | object:
        if self.vel is _UNSET:
            return _UNSET
        return self.vel[1]

    def plot(self, ax: plt.Axes) -> None:
        r"""
        draw the particle position on a matplotlib axes.

        the marker area scales like `s \propto \sqrt{m}` so heavier particles
        remain visibly larger without dominating the frame.
        """
        ax.scatter([self.x()], [self.y()], s=10, color=self.color)

    def distance_to(self, coords2: np.ndarray, box: PeriodicSquareBox2D) -> float:
        """
        return the minimum-image distance to another position in the box.

        this is the separation that would enter the gravitational interaction
        with another particle at `coords2`.
        """
        return box.distance(self.coords, coords2)

    def __repr__(self) -> str:
        vel = "<unset>" if self.vel is _UNSET else self.vel
        return (
            f"Particle with type={self.type} @ t={self.time}, x={self.coords}, v={vel}"
        )


class ParticleLifeInteraction:
    r"""
    pairwise particle-life interaction with a fixed type-to-type force matrix.

    for particle types `i` and `j`, the interaction strength is `a_{ij}` from a
    matrix sampled once at initialization. the pair force is
    `f_{i \leftarrow j} = s \, \psi_{ij}(r) \, \hat{r}`,
    where `s` is a global force scale,
    `r = \|x_j - x_i\|`,
    and `\hat{r} = (x_j - x_i) / r`.

    the radial law is split into an unconditional repulsive core and a
    type-dependent interaction shell:

    `\psi_{ij}(r) = r / r_{rep} - 1` for `r < r_{rep}`,

    `\psi_{ij}(r) = a_{ij} \left[1 - (r - r_{rep}) / (r_{int} - r_{rep})\right]`
    for `r_{rep} \le r < r_{int}`,

    and `\psi_{ij}(r) = 0` for `r \ge r_{int}`.
    """

    def __init__(
        self,
        n_types: int,
        interaction_radius: float = 6.0,
        repulsion_radius: float = 3.0,
        force_scale: float = 1.0,
        seed: int | None = None,
        interaction_matrix: np.ndarray | None = None,
    ) -> None:
        if n_types <= 0:
            raise ValueError("n_types must be positive.")
        if interaction_radius <= 0:
            raise ValueError("interaction_radius must be positive.")
        if repulsion_radius <= 0:
            raise ValueError("repulsion_radius must be positive.")
        if repulsion_radius >= interaction_radius:
            raise ValueError("repulsion_radius must be smaller than interaction_radius.")

        self.n_types = n_types
        self.interaction_radius = interaction_radius
        self.repulsion_radius = repulsion_radius
        self.force_scale = force_scale

        if interaction_matrix is None:
            rng = np.random.default_rng(seed)
            self.interaction_matrix = rng.uniform(-1.0, 1.0, size=(n_types, n_types))
        else:
            matrix = np.asarray(interaction_matrix, dtype=float)
            if matrix.shape != (n_types, n_types):
                raise ValueError(
                    f"interaction_matrix must have shape {(n_types, n_types)}."
                )
            self.interaction_matrix = matrix

    def interaction_strength(self, particle_1: Particle, particle_2: Particle) -> float:
        """
        return the fixed type-to-type coefficient `a_{ij}`.
        """
        return self.interaction_matrix[particle_1.type, particle_2.type]

    def radial_kernel(self, distance: float) -> float:
        r"""
        return the scalar type-dependent shell factor `\phi(r)`.

        for `r_{rep} \le r < r_{int}`, the interaction magnitude decays linearly:
        `\phi(r) = 1 - (r - r_{rep}) / (r_{int} - r_{rep})`.

        for `r < r_{rep}` and for `r \ge r_{int}`, this shell factor is not used.

        for `r \ge r_{int}`, the interaction vanishes:
        `\phi(r) = 0`.
        """
        if distance >= self.interaction_radius:
            return 0.0
        return 1.0 - (
            (distance - self.repulsion_radius)
            / (self.interaction_radius - self.repulsion_radius)
        )

    def force(
        self,
        particle_1: Particle,
        particle_2: Particle,
        box: PeriodicSquareBox2D,
    ) -> np.ndarray:
        r"""
        force acting on particle 1 due to particle 2.

        the implemented law is
        `f_{1 \leftarrow 2} = s \, (r / r_{rep} - 1) \, \hat{r}` for
        `r < r_{rep}`,

        and
        `f_{1 \leftarrow 2} = s \, a_{12} \, \phi(r) \, \hat{r}` for
        `r_{rep} \le r < r_{int}`,

        with `\hat{r} = (x_2 - x_1) / \|x_2 - x_1\|`.
        """
        distance, displacement = box.distance(
            particle_1.coords, particle_2.coords, with_disp=True
        )
        if distance == 0:
            return np.zeros(2, dtype=float)

        direction = displacement / distance
        if distance < self.repulsion_radius:
            radial_factor = distance / self.repulsion_radius - 1.0
            return self.force_scale * radial_factor * direction

        strength = self.interaction_strength(particle_1, particle_2)
        radial_factor = self.radial_kernel(distance)
        return self.force_scale * strength * radial_factor * direction


def copy_particles(particles: list[Particle]) -> list[Particle]:
    """
    make detached particle copies so trajectories and previews do not alias.
    """
    return [
        Particle(
            mass=particle.mass,
            time=particle.time,
            coords=np.array(particle.coords, copy=True),
            vel=_UNSET
            if particle.vel is _UNSET
            else np.array(particle.vel, copy=True),
            type_=particle.type,
        )
        for particle in particles
    ]


def particle_arrays(
    particles: list[Particle],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    pack particle objects into dense arrays for numerical kernels.
    """
    coords = np.array([particle.coords for particle in particles], dtype=float)
    velocities = np.array([particle.vel for particle in particles], dtype=float)
    masses = np.array([particle.mass for particle in particles], dtype=float)
    times = np.array([particle.time for particle in particles], dtype=float)
    types = np.array([particle.type for particle in particles], dtype=np.int64)
    return coords, velocities, masses, times, types


def particles_from_arrays(
    coords: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    times: np.ndarray,
    types: np.ndarray,
) -> list[Particle]:
    """
    rebuild particle objects from dense arrays after a numerical update.
    """
    return [
        Particle(
            mass=float(masses[i]),
            time=float(times[i]),
            coords=np.array(coords[i], copy=True),
            vel=np.array(velocities[i], copy=True),
            type_=int(types[i]),
        )
        for i in range(coords.shape[0])
    ]


def make_logger(log_path: Path) -> logging.Logger:
    """
    create the script logger with sysentropy.
    """
    logger_config = LoggerConfig(
        level=logging.INFO,
        log_file=log_path,
        console_format="%(asctime)s %(levelname_fixed)s %(message)s",
        file_format="%(asctime)s %(levelname_fixed)s %(message)s",
        use_colors=True,
    )
    return get_logger("particle-life", config=logger_config)


class Integrator(ABC):
    """
    generic integrator interface in the same spirit as pfnumerics.
    """

    @abstractmethod
    def forward(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        t: float,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        """
        advance the state by one integration step.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        x: Any,
        *,
        all_trj: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        integrate a state or trajectory using a consistent high-level api.
        """
        raise NotImplementedError


class Euler(Integrator):
    r"""
    fixed-step Euler integrator for particle-life dynamics.

    the state update is
    `v_{n+1} = v_n + \Delta t \, (a_n - \gamma v_n)`,
    `x_{n+1} = x_n + \Delta t \, v_n`,
    where `\gamma` is the damping coefficient.
    """

    def __init__(
        self,
        t_bounds: Sequence[float],
        n_steps: int,
        damping_coefficient: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.t_bounds = t_bounds
        self.n_steps = n_steps
        self.dt = (float(t_bounds[1]) - float(t_bounds[0])) / n_steps
        self.damping_coefficient = damping_coefficient
        self.logger = logger

    def forward(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        t: float,
        state: list[Particle],
        **kwargs: Any,
    ) -> list[Particle]:
        """
        advance the particle system by one Euler step.
        """
        box = kwargs["box"]
        accelerations = rhs(state)
        coords, velocities, masses, times, types = particle_arrays(state)
        damped_accelerations = accelerations - self.damping_coefficient * velocities
        next_velocities = velocities + self.dt * damped_accelerations
        next_coords = box.wrap(coords + self.dt * velocities)
        next_times = times + self.dt
        return particles_from_arrays(
            next_coords,
            next_velocities,
            masses,
            next_times,
            types,
        )

    def integrate(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        x: list[Particle],
        *,
        all_trj: bool = False,
        **kwargs: Any,
    ) -> list[list[Particle]] | list[Particle]:
        """
        integrate the particle system across the configured time interval.
        """
        state = copy_particles(x)
        history = [copy_particles(state)] if all_trj else None
        progress_stride = max(1, self.n_steps // 20)

        for step_index in range(self.n_steps):
            t = self.t_bounds[0] + step_index * self.dt
            state = self.forward(rhs, t, state, **kwargs)
            if all_trj:
                history.append(copy_particles(state))
            if self.logger is not None and (
                step_index == 0
                or (step_index + 1) % progress_stride == 0
                or step_index + 1 == self.n_steps
            ):
                self.logger.info(
                    "simulation step %d / %d",
                    step_index + 1,
                    self.n_steps,
                )

        if all_trj:
            return history
        return state


class ParticleLifeSimulation:
    """
    particle-life evolution under pairwise type-dependent interactions.
    """

    def __init__(
        self,
        system: "ParticlesInaBox",
        interaction: ParticleLifeInteraction,
        integrator: Integrator,
        logger: logging.Logger,
        use_cell_list: bool = True,
    ) -> None:
        self.system = system
        self.interaction = interaction
        self.integrator = integrator
        self.logger = logger
        self._uses_numba = njit is not None
        self.use_cell_list = use_cell_list

    def compute_pairwise_forces(self, particles: list[Particle]) -> list[np.ndarray]:
        """
        compute the net particle-life force on each particle.
        """
        coords = np.array([particle.coords for particle in particles], dtype=float)
        types = np.array([particle.type for particle in particles], dtype=np.int64)
        if self.use_cell_list:
            return _compute_particle_life_forces_cell_list(
                coords,
                types,
                self.interaction.interaction_matrix,
                self.system.box.box_length,
                self.interaction.interaction_radius,
                self.interaction.repulsion_radius,
                self.interaction.force_scale,
            )
        return _compute_particle_life_forces(
            coords,
            types,
            self.interaction.interaction_matrix,
            self.system.box.box_length,
            self.interaction.interaction_radius,
            self.interaction.repulsion_radius,
            self.interaction.force_scale,
        )

    def compute_accelerations(self, particles: list[Particle]) -> list[np.ndarray]:
        """
        compute accelerations from the net type-dependent forces.
        """
        forces = self.compute_pairwise_forces(particles)
        return [force / particle.mass for force, particle in zip(forces, particles)]

    def run(self, all_trj: bool = True) -> list[list[Particle]] | list[Particle]:
        """
        run the particle-life simulation with the configured integrator.
        """
        self.logger.info("running simulation with %s", type(self.integrator).__name__)
        self.logger.info("numba acceleration enabled: %s", self._uses_numba)
        self.logger.info("cell list enabled: %s", self.use_cell_list)
        result = self.integrator.integrate(
            self.compute_accelerations,
            copy_particles(self.system.particles),
            box=self.system.box,
            all_trj=all_trj,
        )
        final_particles = result[-1] if all_trj else result
        self.system.particles = copy_particles(final_particles)
        self.logger.info("finished simulation")
        return result

    def live_preview(
        self,
        *,
        steps_per_frame: int = 4,
        interval_ms: int = 16,
    ) -> None:
        """
        render the evolving particle-life system in a live matplotlib window.
        """
        if steps_per_frame <= 0:
            raise ValueError("steps_per_frame must be positive.")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive.")
        if not isinstance(self.integrator, Euler):
            raise TypeError("live_preview currently expects an Euler integrator.")

        self.logger.info(
            "starting live preview with steps_per_frame=%d",
            steps_per_frame,
        )

        box = self.system.box
        coords, velocities, masses, times, types = particle_arrays(self.system.particles)
        current_time = float(times[0])
        point_colors = [TYPE_MAP[int(type_index)] for type_index in types]
        point_sizes = np.full(coords.shape[0], 9.0, dtype=float)

        figure, ax = fm.layouts.subplots(width="single", aspect=1.0)
        fm.clean_axes(ax, keep=("left", "bottom"), grid=False, legend=False)
        ax.set_xlim(0, box.box_length)
        ax.set_ylim(0, box.box_length)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_sizes,
            c=point_colors,
            edgecolors="none",
            animated=True,
        )

        def update(_frame_index: int):
            nonlocal current_time

            _advance_particle_life_euler_inplace(
                coords,
                velocities,
                masses,
                types,
                self.interaction.interaction_matrix,
                box.box_length,
                self.interaction.interaction_radius,
                self.interaction.repulsion_radius,
                self.interaction.force_scale,
                self.integrator.damping_coefficient,
                self.integrator.dt,
                steps_per_frame,
            )
            current_time += steps_per_frame * self.integrator.dt
            scatter.set_offsets(coords)
            return (scatter,)

        animation = FuncAnimation(
            figure,
            update,
            interval=interval_ms,
            blit=True,
            cache_frame_data=False,
        )
        figure._live_preview_animation = animation
        plt.show()
        final_times = np.full(coords.shape[0], current_time, dtype=float)
        self.system.particles = particles_from_arrays(
            coords,
            velocities,
            masses,
            final_times,
            types,
        )
        self.logger.info("live preview closed")


class ParticlesInaBox:
    """
    collection of particles embedded in a periodic square box.
    """

    def __init__(
        self,
        particles: list[Particle] | None = None,
        box: PeriodicSquareBox2D | None = None,
    ) -> None:
        self.particles = list(particles) if particles is not None else []
        self.box = box
        if self.box is not None:
            for particle in self.particles:
                particle.coords = self.box.wrap(particle.coords)

    def print_particles(self) -> None:
        for p in self.particles:
            print(p)

    def plot(self, ax: plt.Axes) -> None:
        """
        draw all particles in the box on a matplotlib axes.
        """
        for particle in self.particles:
            particle.plot(ax)

    def live_preview(self) -> None:
        """
        show the current particle configuration in an interactive matplotlib window.
        """
        figure, ax = fm.layouts.subplots(width="single", aspect=1.0)
        self.plot(ax)
        ax.set_xlim(0, self.box.box_length)
        ax.set_ylim(0, self.box.box_length)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        fm.clean_axes(ax, keep=("left", "bottom"), grid=True, legend=False)
        figure.tight_layout()
        plt.show()

    @classmethod
    def build_initial(
        cls,
        number_particles: int,
        box: PeriodicSquareBox2D,
        style: str = "random",
        seed: int | None = None,
        type_probabilities: np.ndarray | None = None,
        velocity_scale: float = 0.0,
        extra_random_particles: int = 0,
        extra_type_probabilities: np.ndarray | None = None,
    ) -> "ParticlesInaBox":
        """
        build an initial condition inside the periodic box.

        for `style = "random"`, positions are sampled uniformly in the box.

        for `style = "core_shell"`, type `1` is seeded in a central disk, type
        `0` is seeded in an annulus around it, and higher types are seeded in
        progressively broader outer bands. this gives each species a distinct
        initial radial role instead of asking the morphology to emerge from a
        fully homogeneous cloud.

        if `extra_random_particles > 0`, additional particles are then sampled
        uniformly in the box. this lets the simulation keep a seeded core/shell
        droplet while also adding a dilute background population.
        """
        if extra_random_particles < 0:
            raise ValueError("extra_random_particles must be non-negative.")

        types = TYPE_MAP.keys()
        rng = np.random.default_rng(seed)
        if type_probabilities is None:
            type_probabilities = np.full(len(types), 1.0 / len(types), dtype=float)
        else:
            type_probabilities = np.asarray(type_probabilities, dtype=float)
            if type_probabilities.shape != (len(types),):
                raise ValueError(
                    f"type_probabilities must have shape {(len(types),)}."
                )
            total = np.sum(type_probabilities)
            if total <= 0:
                raise ValueError("type_probabilities must have positive sum.")
            type_probabilities = type_probabilities / total

        if extra_type_probabilities is None:
            extra_type_probabilities = type_probabilities
        else:
            extra_type_probabilities = np.asarray(extra_type_probabilities, dtype=float)
            if extra_type_probabilities.shape != (len(types),):
                raise ValueError(
                    f"extra_type_probabilities must have shape {(len(types),)}."
                )
            total = np.sum(extra_type_probabilities)
            if total <= 0:
                raise ValueError("extra_type_probabilities must have positive sum.")
            extra_type_probabilities = extra_type_probabilities / total

        if style == "random":
            particles = [
                Particle(
                    mass=1.0,
                    time=0,
                    coords=box.random_coordinate(),
                    vel=rng.normal(loc=0.0, scale=velocity_scale, size=2),
                    type_=rng.choice(len(types), p=type_probabilities),
                )
                for _ in range(number_particles)
            ]
        elif style == "core_shell":
            center = np.full(2, box.box_length / 2.0, dtype=float)
            radial_means = np.array([0.22, 0.08, 0.34, 0.43, 0.50], dtype=float)
            radial_widths = np.array([0.035, 0.035, 0.050, 0.045, 0.035], dtype=float)
            particles = []

            for _ in range(number_particles):
                type_index = int(rng.choice(len(types), p=type_probabilities))
                angle = rng.uniform(0.0, 2.0 * np.pi)

                if type_index == 1:
                    radius = abs(rng.normal(scale=0.035 * box.box_length))
                else:
                    mean_index = min(type_index, len(radial_means) - 1)
                    radius = box.box_length * radial_means[mean_index]
                    radius += rng.normal(
                        scale=box.box_length * radial_widths[mean_index]
                    )
                    radius = max(0.0, radius)

                coords = center + radius * np.array(
                    [np.cos(angle), np.sin(angle)],
                    dtype=float,
                )
                particles.append(
                    Particle(
                        mass=1.0,
                        time=0,
                        coords=box.wrap(coords),
                        vel=rng.normal(loc=0.0, scale=velocity_scale, size=2),
                        type_=type_index,
                    )
                )
        else:
            raise ValueError(
                f"Unrecognized style {style}. Supported 'random', 'core_shell'"
            )

        for _ in range(extra_random_particles):
            particles.append(
                Particle(
                    mass=1.0,
                    time=0,
                    coords=box.random_coordinate(),
                    vel=rng.normal(loc=0.0, scale=velocity_scale, size=2),
                    type_=int(rng.choice(len(types), p=extra_type_probabilities)),
                )
            )

        return ParticlesInaBox(particles, box)

    def __repr__(self) -> str:
        return f"System with {len(self.particles)} particles in box of size {self.box.box_length}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show the current particle configuration in an interactive window",
    )
    parser.add_argument(
        "--state-seed",
        type=int,
        default=5,
        help="seed for the initial particle positions and velocities",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    logger = make_logger(project_dir / "particle-life.log")
    number_particles = 700
    box = PeriodicSquareBox2D(box_length=20)
    type_probabilities = np.array([0.30, 0.24, 0.18, 0.16, 0.12], dtype=float)
    system = ParticlesInaBox.build_initial(
        number_particles,
        box,
        style="core_shell",
        seed=args.state_seed,
        type_probabilities=type_probabilities,
        velocity_scale=0.0,
    )
    interaction_matrix = np.array(
        [
            [0.18, 0.36, 0.14, 0.08, -0.10],
            [-0.28, 0.56, -0.10, -0.18, -0.26],
            [0.28, -0.16, 0.16, 0.24, 0.06],
            [0.14, -0.24, 0.18, 0.20, 0.28],
            [-0.04, -0.30, 0.10, 0.24, 0.12],
        ],
        dtype=float,
    )
    interaction = ParticleLifeInteraction(
        n_types=len(TYPE_MAP),
        interaction_radius=5.6,
        repulsion_radius=1.2,
        force_scale=0.95,
        interaction_matrix=interaction_matrix,
    )
    integrator = Euler(
        t_bounds=(0.0, 500.0),
        n_steps=200_000,
        damping_coefficient=0.85,
        logger=logger,
    )
    simulation = ParticleLifeSimulation(
        system=system,
        interaction=interaction,
        integrator=integrator,
        logger=logger,
    )
    if args.preview:
        fm.apply_style(
            {
                "figure.figsize": fm.layouts.size(width="single", aspect=1.0),
                "axes.grid": False,
            }
        )
        logger.info(
            "preview with state_seed=%d",
            args.state_seed,
        )
        simulation.live_preview(steps_per_frame=16, interval_ms=16)
    else:
        system.print_particles()
        print(system)
        print(f"state_seed={args.state_seed}")
        print(interaction.interaction_matrix)


if __name__ == "__main__":
    main()
