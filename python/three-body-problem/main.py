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

# sentinel object
_UNSET = object()


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
    ) -> None:

        self.time = time
        self.mass = mass
        self.coords = np.asarray(coords, dtype=float)
        self.vel = _UNSET if vel is _UNSET else np.asarray(vel, dtype=float)

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
        ax.scatter([self.x()], [self.y()], s=20 * np.sqrt(self.mass))

    def distance_to(self, coords2: np.ndarray, box: PeriodicSquareBox2D) -> float:
        """
        return the minimum-image distance to another position in the box.

        this is the separation that would enter the gravitational interaction
        with another particle at `coords2`.
        """
        return box.distance(self.coords, coords2)

    def __repr__(self) -> str:
        vel = "<unset>" if self.vel is _UNSET else self.vel
        return f"Particle with m={self.mass} @ t={self.time}, x={self.coords}, v={vel}"


class ParticlesInaBox(list):
    """
    collection of particles embedded in a periodic square box.

    the particles are assumed to interact through pairwise gravitational forces
    computed with the box minimum-image convention.
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

    def __repr__(self) -> str:
        return f"System with {len(self.particles)} particles in box of size {self.box.box_length}"


def copy_particles(particles: list[Particle]) -> list[Particle]:
    """
    make detached particle copies for trajectory storage and stepping.

    this avoids aliasing, i.e. later updates should not mutate an already
    saved state `x_n`.
    """
    return [
        Particle(
            mass=particle.mass,
            time=particle.time,
            coords=np.array(particle.coords, copy=True),
            vel=_UNSET if particle.vel is _UNSET else np.array(particle.vel, copy=True),
        )
        for particle in particles
    ]


def make_logger(log_path: Path) -> logging.Logger:
    logger_config = LoggerConfig(
        level=logging.INFO,
        log_file=log_path,
        console_format="%(asctime)s %(levelname_fixed)s %(message)s",
        file_format="%(asctime)s %(levelname_fixed)s %(message)s",
        use_colors=True,
    )
    return get_logger("three-body-problem", config=logger_config)


class GravitationalInteraction:
    """
    pairwise newtonian gravitational interaction.

    the potential is `u(r) = -g m_1 m_2 / r`, and the force is the associated
    attractive inverse-square gravitational force.
    """

    def __init__(self, gravitational_constant: float = 1) -> None:
        self.gravitational_constant = gravitational_constant

    def potential_prefactor(self, particle_1: Particle, particle_2: Particle) -> float:
        r"""
        return the mass-dependent prefactor of the gravitational potential.

        `c = -g m_1 m_2`.
        """
        return -self.gravitational_constant * particle_1.mass * particle_2.mass

    def potential_energy(
        self,
        particle_1: Particle,
        particle_2: Particle,
        box: PeriodicSquareBox2D,
    ) -> float:
        r"""
        return the pair gravitational potential energy.
        """
        return self.potential_prefactor(particle_1, particle_2) / box.distance(
            particle_1.coords, particle_2.coords
        )

    def force(
        self,
        particle_1: Particle,
        particle_2: Particle,
        box: PeriodicSquareBox2D,
    ) -> np.ndarray:
        r"""
        force acting on particle 1 due to particle 2.
        """

        # the displacement order matters here because the sign of the force is
        # encoded by the direction of `dr`
        scale = self.potential_prefactor(particle_2, particle_1)
        distance, dr = box.distance(
            particle_2.coords, particle_1.coords, with_disp=True
        )
        return (scale / distance**3) * dr


@dataclass
class VerletState:
    r"""
    verlet state carrying the current and previous particle slices.

    position-verlet evolves from `(x_{n-1}, x_n)` to `x_{n+1}`, so both time
    slices are part of the integrator state.
    """

    previous_particles: list[Particle]
    current_particles: list[Particle]


class Integrator(ABC):
    r"""
    generic integrator interface

    the abstract idea is that an integrator advances a state by repeatedly
    applying a rule of the form
    `x_{n+1} = \phi(x_n, t_n)`.
    """

    @abstractmethod
    def forward(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        t: float,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        r"""
        advance the state by one integration step.

        conceptually this returns something like
        `x_{n+1} = \phi(x_n, t_n)`.
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
        r"""
        integrate a state or trajectory using a consistent high-level api.

        when `all_trj = \mathrm{true}`, the return value should contain the
        whole sampled trajectory `(x_0, x_1, \ldots, x_n)`.
        """
        raise NotImplementedError


class PositionVerlet(Integrator):
    r"""
    fixed-step position-verlet integrator for particle systems.

    this integrator evolves particle positions under a supplied acceleration
    operator. initial velocities are used only to construct the previous-time
    slice needed to start the verlet recurrence.

    the recurrence is
    `x_{n+1} = 2 x_n - x_{n-1} + a_n \Delta t^2`,
    while the initial step is simple parabolic motion
    `x_{-1} = x_0 - v_0 \Delta t + \tfrac{1}{2} a_0 \Delta t^2`.
    """

    def __init__(
        self,
        t_bounds: Sequence[float],
        n_steps: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.t_bounds = t_bounds
        self.n_steps = n_steps
        self.dt = (float(t_bounds[1]) - float(t_bounds[0])) / n_steps
        self.logger = logger

    def initialize(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        particles: list[Particle],
        *,
        box: PeriodicSquareBox2D,
    ) -> VerletState:
        r"""
        build the initial verlet state from the provided particle slice.

        this uses the bootstrap formula
        `x_{-1} = x_0 - v_0 \Delta t + \tfrac{1}{2} a_0 \Delta t^2`
        so that subsequent steps can use the pure position recurrence.
        """
        accelerations = rhs(particles)
        previous_particles = []

        for particle, acceleration in zip(particles, accelerations):
            if particle.vel is _UNSET:
                raise ValueError(
                    "Initial particles must have velocities set to start Verlet integration."
                )

            # the previous slice is reconstructed from the given initial
            # position, velocity, and acceleration
            previous_coords = (
                particle.coords
                - particle.vel * self.dt
                + 0.5 * acceleration * self.dt**2
            )
            previous_coords = box.wrap(previous_coords)
            previous_particles.append(
                Particle(
                    mass=particle.mass,
                    time=particle.time - self.dt,
                    coords=previous_coords,
                    vel=_UNSET,
                )
            )

        return VerletState(
            previous_particles=previous_particles,
            current_particles=copy_particles(particles),
        )

    def forward(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        t: float,
        state: VerletState,
        **kwargs: Any,
    ) -> VerletState:
        r"""
        advance the verlet state by one step.

        the update is
        `x_{n+1} = x_n + (x_n - x_{n-1}) + a_n \Delta t^2`,
        which is algebraically the same as
        `x_{n+1} = 2 x_n - x_{n-1} + a_n \Delta t^2`.
        """
        box = kwargs["box"]
        accelerations = rhs(state.current_particles)
        next_particles = []

        for previous_particle, current_particle, acceleration in zip(
            state.previous_particles, state.current_particles, accelerations
        ):
            # in a periodic box we do not want the raw wrapped coordinates
            # `x_n` and `x_{n-1}` to create a fake long jump across the whole
            # box, so we reconstruct the actual short step using the
            # minimum-image displacement first
            step_displacement = box.displacement(
                previous_particle.coords, current_particle.coords
            )
            next_coords = (
                current_particle.coords + step_displacement + acceleration * self.dt**2
            )
            next_coords = box.wrap(next_coords)
            next_particles.append(
                Particle(
                    mass=current_particle.mass,
                    time=current_particle.time + self.dt,
                    coords=next_coords,
                    vel=_UNSET,
                )
            )

        return VerletState(
            previous_particles=copy_particles(state.current_particles),
            current_particles=next_particles,
        )

    def integrate(
        self,
        rhs: Callable[[list[Particle]], list[np.ndarray]],
        x: list[Particle],
        *,
        all_trj: bool = False,
        **kwargs: Any,
    ) -> list[list[Particle]] | list[Particle]:
        r"""
        integrate the particle system across the configured time interval.

        if `all_trj = \mathrm{true}`, this returns the sampled trajectory
        `(x_0, x_1, \ldots, x_n)`. otherwise it returns only the final slice
        `x_n`.
        """
        box = kwargs["box"]
        state = self.initialize(rhs, x, box=box)
        history = [copy_particles(x)] if all_trj else None
        progress_stride = max(1, self.n_steps // 20)

        for step_index in range(self.n_steps):
            t = self.t_bounds[0] + step_index * self.dt
            state = self.forward(rhs, t, state, box=box)
            if all_trj:
                history.append(copy_particles(state.current_particles))
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
        return state.current_particles


class ThreeBodySimulation:
    r"""
    three-body evolution under pairwise gravitational interactions.

    the positions are advanced with position-verlet integration. the initial
    velocities are used only to construct the previous-time positions needed to
    start the verlet recurrence.

    the system dynamics are
    `m_i \ddot{x}_i = \sum_{j \neq i} f_{i \leftarrow j}`.
    """

    def __init__(
        self,
        system: ParticlesInaBox,
        interaction: GravitationalInteraction,
        integrator: Integrator,
        logger: logging.Logger,
    ) -> None:
        self.system = system
        self.interaction = interaction
        self.integrator = integrator
        self.logger = logger

    def compute_pairwise_forces(self, particles: list[Particle]) -> list[np.ndarray]:
        r"""
        compute the net gravitational force on each particle.
        """
        forces = [np.zeros(2, dtype=float) for _ in particles]

        for i, particle_1 in enumerate(particles):
            for j in range(i + 1, len(particles)):
                particle_2 = particles[j]
                # each pair contributes equal and opposite forces, so we can
                # accumulate both particles in one pass
                force_ij = self.interaction.force(
                    particle_1, particle_2, self.system.box
                )
                forces[i] += force_ij
                forces[j] -= force_ij

        return forces

    def compute_accelerations(self, particles: list[Particle]) -> list[np.ndarray]:

        forces = self.compute_pairwise_forces(particles)
        return [force / particle.mass for force, particle in zip(forces, particles)]

    def run(self, all_trj: bool = True) -> list[list[Particle]] | list[Particle]:
        r"""
        run the gravitational three-body simulation with the configured integrator.

        when `all_trj = \mathrm{true}`, the returned object is the sampled
        trajectory `(x_0, x_1, \ldots, x_n)`.
        """
        self.logger.info("running simulation with %s", type(self.integrator).__name__)
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

    def save_frames(
        self,
        history: list[list[Particle]],
        output_dir: str | Path = "frames",
        trail_length: int = 150,
        frame_stride: int = 1,
    ) -> None:
        r"""
        save one rendered frame per stored simulation state.

        each frame shows the particle positions in the periodic box for the
        gravitational three-body evolution, together with a recent trajectory
        trail for each particle.

        if the saved trajectory is `(x_0, x_1, \ldots, x_n)`, then each frame
        visualizes a local window
        `(x_{k-\ell+1}, \ldots, x_k)`
        with `\ell = \mathrm{trail\_length}`.
        """
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        trail_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        selected_steps = list(range(0, len(history), frame_stride))
        progress_stride = max(1, len(selected_steps) // 20)

        self.logger.info(
            "saving %d frame images to %s with frame_stride=%d",
            len(selected_steps),
            output_path,
            frame_stride,
        )

        for frame_index, step in enumerate(selected_steps):
            particles = history[step]
            figure, ax = fm.layouts.subplots(width="single", aspect=1.0)
            particle_snapshot = ParticlesInaBox(
                particles=particles, box=self.system.box
            )

            trail_start = max(0, step - trail_length + 1)
            trail_history = history[trail_start : step + 1]
            for particle_index in range(len(particles)):
                trail_coords = np.array(
                    [snapshot[particle_index].coords for snapshot in trail_history]
                )
                trail_coords = np.asarray(trail_coords, dtype=float)
                trail_jumps = np.abs(np.diff(trail_coords, axis=0)) > (
                    0.5 * self.system.box.box_length
                )
                if np.any(trail_jumps):
                    # a jump larger than half the box almost certainly means
                    # the particle crossed the periodic seam, so we break the
                    # rendered line instead of drawing an artificial diagonal
                    trail_coords = trail_coords.copy()
                    trail_coords[1:][trail_jumps.any(axis=1)] = np.nan

                ax.plot(
                    trail_coords[:, 0],
                    trail_coords[:, 1],
                    color=trail_colors[particle_index % len(trail_colors)],
                    alpha=0.4,
                    linewidth=1.5,
                )

            particle_snapshot.plot(ax)
            ax.set_xlim(0, self.system.box.box_length)
            ax.set_ylim(0, self.system.box.box_length)
            ax.set_aspect("equal")
            ax.set_title(rf"$n = {step},\ t = {particles[0].time:.3f}$")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            fm.clean_axes(ax, keep=("left", "bottom"), grid=True, legend=False)
            figure.tight_layout()
            figure.savefig(output_path / f"frame_{frame_index:04d}.png")
            plt.close(figure)
            if (
                frame_index == 0
                or (frame_index + 1) % progress_stride == 0
                or frame_index + 1 == len(selected_steps)
            ):
                self.logger.info(
                    "frame %d / %d",
                    frame_index + 1,
                    len(selected_steps),
                )

        self.logger.info("finished writing frame images")

    def live_preview(
        self,
        *,
        steps_per_frame: int = 8,
        trail_length: int = 250,
        interval_ms: int = 16,
    ) -> None:
        r"""
        render the simulation in a live matplotlib window without saving frames.

        the displayed state advances by repeatedly applying the integrator
        update, while each particle keeps a recent trail
        `(x_{k-\ell+1}, \ldots, x_k)`
        with `\ell = \mathrm{trail\_length}`.
        """
        if steps_per_frame <= 0:
            raise ValueError("steps_per_frame must be positive.")
        if trail_length <= 0:
            raise ValueError("trail_length must be positive.")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive.")
        if not isinstance(self.integrator, PositionVerlet):
            raise TypeError("live_preview currently expects a PositionVerlet integrator.")

        self.logger.info(
            "starting live preview with steps_per_frame=%d, trail_length=%d",
            steps_per_frame,
            trail_length,
        )

        box = self.system.box
        state = self.integrator.initialize(
            self.compute_accelerations,
            copy_particles(self.system.particles),
            box=box,
        )
        current_time = state.current_particles[0].time

        figure, ax = fm.layouts.subplots(width="single", aspect=1.0)
        fm.clean_axes(ax, keep=("left", "bottom"), grid=True, legend=False)
        ax.set_xlim(0, box.box_length)
        ax.set_ylim(0, box.box_length)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        trail_buffers = [
            deque([np.array(particle.coords, copy=True)], maxlen=trail_length)
            for particle in state.current_particles
        ]
        trail_lines = []
        point_artists = []

        for particle_index, particle in enumerate(state.current_particles):
            color = colors[particle_index % len(colors)]
            (trail_line,) = ax.plot(
                [],
                [],
                color=color,
                alpha=0.4,
                linewidth=1.5,
            )
            (point_artist,) = ax.plot(
                [particle.x()],
                [particle.y()],
                marker="o",
                linestyle="none",
                color=color,
                # markersize=4.0 * np.sqrt(particle.mass),
                markersize=4.0,
            )
            trail_lines.append(trail_line)
            point_artists.append(point_artist)

        def append_trail(buffer: deque[np.ndarray], coords: np.ndarray) -> None:
            # if the particle crosses the periodic seam, insert a nan marker so
            # matplotlib breaks the rendered line instead of drawing across the box
            if buffer:
                previous_coords = buffer[-1]
                if np.any(np.abs(coords - previous_coords) > 0.5 * box.box_length):
                    buffer.append(np.array([np.nan, np.nan]))
            buffer.append(np.array(coords, copy=True))

        def update(_frame_index: int):
            nonlocal state, current_time

            for _ in range(steps_per_frame):
                state = self.integrator.forward(
                    self.compute_accelerations,
                    current_time,
                    state,
                    box=box,
                )
                current_time = state.current_particles[0].time
                for particle_index, particle in enumerate(state.current_particles):
                    append_trail(trail_buffers[particle_index], particle.coords)

            for particle_index, particle in enumerate(state.current_particles):
                trail_coords = np.asarray(trail_buffers[particle_index], dtype=float)
                trail_lines[particle_index].set_data(
                    trail_coords[:, 0],
                    trail_coords[:, 1],
                )
                point_artists[particle_index].set_data(
                    [particle.x()],
                    [particle.y()],
                )

            return trail_lines + point_artists

        animation = FuncAnimation(
            figure,
            update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        figure._live_preview_animation = animation
        plt.show()
        self.system.particles = copy_particles(state.current_particles)
        self.logger.info("live preview closed")

    def make_video(
        self,
        frames_dir: str | Path,
        output_file: str | Path,
        frame_rate: int = 30,
    ) -> None:
        r"""
        assemble the saved frame images into an mp4 using ffmpeg.

        the ffmpeg filter pads the raster to an even size,
        `w' = 2 \lceil w / 2 \rceil`,
        `h' = 2 \lceil h / 2 \rceil`,
        because `libx264` expects dimensions divisible by 2.
        """
        frames_path = Path(frames_dir)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(frame_rate),
            "-i",
            str(frames_path / "frame_%04d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self.logger.info(
            "assembling mp4 from %s to %s at %d fps",
            frames_path,
            output_path,
            frame_rate,
        )
        try:
            subprocess.run(ffmpeg_command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to create the mp4 output.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("ffmpeg failed while creating the mp4 output.") from exc
        self.logger.info("finished writing mp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-preview",
        action="store_true",
        help="render the simulation in a live window instead of saving frames",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    logger = make_logger(project_dir / "three-body-problem.log")
    fm.apply_style(
        {
            "figure.figsize": fm.layouts.size(width="single", aspect=1.0),
            "axes.grid": True,
        }
    )
    box = PeriodicSquareBox2D(box_length=100)
    box_center = np.array([box.half_box_length, box.half_box_length])

    # this is a deliberately asymmetric binary-plus-intruder setup.
    # it is less orderly than the figure-eight orbit, which tends to make the
    # trails and the final video more visually interesting.
    r1 = box_center + np.array([0.0, 7.0])
    r2 = box_center
    r3 = box_center + np.array([0.0, -20.0])

    v1 = np.array([-1.0, -0.1])
    v2 = np.array([0.0, 0.0])
    v3 = np.array([1.5, 0.2])

    p1 = Particle(mass=1.0, time=0, coords=r1, vel=v1)
    p2 = Particle(mass=100, time=0, coords=r2, vel=v2)
    p3 = Particle(mass=1.0, time=0, coords=r3, vel=v3)
    system = ParticlesInaBox([p1, p2, p3], box=box)

    interaction = GravitationalInteraction(gravitational_constant=0.5)
    integrator = PositionVerlet(
        t_bounds=(0.0, 144.0),
        n_steps=24_000,
        logger=logger,
    )
    simulation = ThreeBodySimulation(
        system, interaction, integrator=integrator, logger=logger
    )
    if args.live_preview:
        simulation.live_preview(
            steps_per_frame=25,
            trail_length=500,
            interval_ms=16,
        )
    else:
        history = simulation.run(all_trj=True)
        output_dir = project_dir / "frames"
        video_path = project_dir / "three-body-problem.mp4"
        simulation.save_frames(
            history,
            output_dir=output_dir,
            trail_length=160,
            frame_stride=14,
        )
        simulation.make_video(output_dir, video_path)


if __name__ == "__main__":
    main()
