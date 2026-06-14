from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as plt
import fubumio as fm
from sysentropy import LoggerConfig, get_logger

_UNSET = object()


@dataclass
class PeriodicSquareBox2D:
    """
    2d square simulation box with periodic boundary conditions.

    this box provides the minimum-image geometry used for pairwise
    gravitational interactions between particles.
    """

    box_length: float

    def __post_init__(self) -> None:
        self.half_box_length = self.box_length / 2

    def wrap(self, coords: np.ndarray) -> np.ndarray:
        """
        wrap coordinates into [0, l).
        """
        wrapped = np.mod(coords, self.box_length)
        return np.where(np.isclose(wrapped, self.box_length), 0.0, wrapped)

    def displacement(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        minimum-image displacement vector r2 - r1.

        this displacement is the one used when evaluating gravitational forces
        in the periodic box.
        """
        dr = np.asarray(coords2) - np.asarray(coords1)
        dr = dr - self.box_length * np.round(dr / self.box_length)
        return dr

    def distance(
        self, coords1: np.ndarray, coords2: np.ndarray, with_disp: bool = False
    ) -> float | tuple[float, np.ndarray]:
        """
        minimum-image distance.

        this distance is the scalar separation entering the pairwise
        gravitational interaction.
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

    this particle is intended for use in a pairwise newtonian gravitational
    interaction.
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
        """
        draw the particle position on a matplotlib axes.
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
        """
        return the mass-dependent prefactor of the gravitational potential.
        """
        return -self.gravitational_constant * particle_1.mass * particle_2.mass

    def potential_energy(
        self,
        particle_1: Particle,
        particle_2: Particle,
        box: PeriodicSquareBox2D,
    ) -> float:
        """
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
        """
        force acting on particle 1 due to particle 2.

        this is the pairwise gravitational force used in the three-body
        evolution.
        """

        # the displacement order is chosen so that dr points from particle 1 to particle 2
        scale = self.potential_prefactor(particle_2, particle_1)
        distance, dr = box.distance(
            particle_2.coords, particle_1.coords, with_disp=True
        )
        return (scale / distance**3) * dr


class ThreeBodySimulation:
    """
    three-body evolution under pairwise gravitational interactions.

    the positions are advanced with position-verlet integration. the initial
    velocities are used only to construct the previous-time positions needed to
    start the verlet recurrence.
    """

    def __init__(
        self,
        system: ParticlesInaBox,
        interaction: GravitationalInteraction,
        time_step: float,
        logger: logging.Logger,
    ) -> None:
        self.system = system
        self.interaction = interaction
        self.time_step = time_step
        self.logger = logger

    def compute_pairwise_forces(self, particles: list[Particle]) -> list[np.ndarray]:
        """
        compute the net gravitational force on each particle.
        """
        forces = [np.zeros(2, dtype=float) for _ in particles]

        for i, particle_1 in enumerate(particles):
            for j in range(i + 1, len(particles)):
                particle_2 = particles[j]
                force_ij = self.interaction.force(
                    particle_1, particle_2, self.system.box
                )
                forces[i] += force_ij
                forces[j] -= force_ij

        return forces

    def compute_accelerations(self, particles: list[Particle]) -> list[np.ndarray]:
        """
        compute the gravitational acceleration on each particle.
        """
        forces = self.compute_pairwise_forces(particles)
        return [force / particle.mass for force, particle in zip(forces, particles)]

    def copy_particles(self, particles: list[Particle]) -> list[Particle]:
        """
        make detached particle copies for trajectory storage.
        """
        return [
            Particle(
                mass=particle.mass,
                time=particle.time,
                coords=np.array(particle.coords, copy=True),
                vel=_UNSET
                if particle.vel is _UNSET
                else np.array(particle.vel, copy=True),
            )
            for particle in particles
        ]

    def initialize_previous_particles(self) -> list[Particle]:
        """
        build the previous-time slice required to start position-verlet.

        for the gravitational three-body problem, this uses the initial
        velocities together with the initial gravitational acceleration.
        """
        accelerations = self.compute_accelerations(self.system.particles)
        previous_particles = []

        for particle, acceleration in zip(self.system.particles, accelerations):
            if particle.vel is _UNSET:
                raise ValueError(
                    "Initial particles must have velocities set to start Verlet integration."
                )

            previous_coords = (
                particle.coords
                - particle.vel * self.time_step
                + 0.5 * acceleration * self.time_step**2
            )
            previous_coords = self.system.box.wrap(previous_coords)
            previous_particles.append(
                Particle(
                    mass=particle.mass,
                    time=particle.time - self.time_step,
                    coords=previous_coords,
                    vel=_UNSET,
                )
            )

        return previous_particles

    def evolve(self, previous_particles: list[Particle]) -> list[Particle]:
        """
        advance the particle positions by one position-verlet step.

        the acceleration comes from the current pairwise gravitational forces.
        velocities are not updated or stored at this stage.
        """
        accelerations = self.compute_accelerations(self.system.particles)
        evolved_particles = []

        for previous_particle, current_particle, acceleration in zip(
            previous_particles, self.system.particles, accelerations
        ):
            # use the minimum-image displacement so verlet remains consistent
            # across periodic boundary wraps
            step_displacement = self.system.box.displacement(
                previous_particle.coords, current_particle.coords
            )
            next_coords = (
                current_particle.coords
                + step_displacement
                + acceleration * self.time_step**2
            )
            next_coords = self.system.box.wrap(next_coords)
            evolved_particles.append(
                Particle(
                    mass=current_particle.mass,
                    time=current_particle.time + self.time_step,
                    coords=next_coords,
                    vel=_UNSET,
                )
            )

        return evolved_particles

    def run(self, n_steps: int) -> list[list[Particle]]:
        """
        run the gravitational three-body simulation for `n_steps` verlet steps.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative.")

        self.logger.info("running simulation for %d verlet steps", n_steps)
        history = [self.copy_particles(self.system.particles)]

        if n_steps == 0:
            return history

        # this constructs the extra time slice needed to start verlet
        previous_particles = self.initialize_previous_particles()
        progress_stride = max(1, n_steps // 20)

        for step_index in range(n_steps):
            next_particles = self.evolve(previous_particles)
            previous_particles = self.system.particles
            self.system.particles = next_particles
            history.append(self.copy_particles(next_particles))
            if (
                step_index == 0
                or (step_index + 1) % progress_stride == 0
                or step_index + 1 == n_steps
            ):
                self.logger.info(
                    "simulation step %d / %d",
                    step_index + 1,
                    n_steps,
                )

        self.logger.info("finished simulation")
        return history

    def save_frames(
        self,
        history: list[list[Particle]],
        output_dir: str | Path = "frames",
        trail_length: int = 150,
        frame_stride: int = 1,
    ) -> None:
        """
        save one rendered frame per stored simulation state.

        each frame shows the particle positions in the periodic box for the
        gravitational three-body evolution, together with a recent trajectory
        trail for each particle.
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
            particle_snapshot = ParticlesInaBox(particles=particles, box=self.system.box)

            trail_start = max(0, step - trail_length + 1)
            trail_history = history[trail_start : step + 1]
            for particle_index in range(len(particles)):
                trail_coords = np.array(
                    [
                        snapshot[particle_index].coords
                        for snapshot in trail_history
                    ]
                )
                trail_coords = np.asarray(trail_coords, dtype=float)
                trail_jumps = np.abs(np.diff(trail_coords, axis=0)) > (
                    0.5 * self.system.box.box_length
                )
                if np.any(trail_jumps):
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
            ax.set_title(
                rf"$n = {step},\ t = {particles[0].time:.3f}$"
            )
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

    def make_video(
        self,
        frames_dir: str | Path,
        output_file: str | Path,
        frame_rate: int = 30,
    ) -> None:
        """
        assemble the saved frame images into an mp4 using ffmpeg.
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
            raise RuntimeError(
                "ffmpeg failed while creating the mp4 output."
            ) from exc
        self.logger.info("finished writing mp4")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    logger = make_logger(project_dir / "three-body-problem.log")
    fm.apply_style(
        {
            "figure.figsize": fm.layouts.size(width="single", aspect=1.0),
            "axes.grid": True,
        }
    )
    box = PeriodicSquareBox2D(box_length=50)
    box_center = np.array([box.half_box_length, box.half_box_length])

    # these are scaled figure-eight initial conditions for the equal-mass
    # three-body problem
    length_scale = 8.0
    velocity_scale = 3.0

    r1 = box_center + length_scale * np.array([-0.97000436, 0.24308753])
    r2 = box_center + length_scale * np.array([0.97000436, -0.24308753])
    r3 = box_center + length_scale * np.array([0.0, 0.0])

    v1 = velocity_scale * np.array([0.4662036850, 0.4323657300])
    v2 = velocity_scale * np.array([0.4662036850, 0.4323657300])
    v3 = velocity_scale * np.array([-0.93240737, -0.86473146])

    p1 = Particle(mass=100, time=0, coords=r1, vel=v1)
    p2 = Particle(mass=1, time=0, coords=r2, vel=v2)
    p3 = Particle(mass=1, time=0, coords=r3, vel=v3)
    system = ParticlesInaBox([p1, p2, p3], box=box)

    interaction = GravitationalInteraction(gravitational_constant=1)
    simulation = ThreeBodySimulation(system, interaction, time_step=0.01, logger=logger)
    history = simulation.run(n_steps=10_000)
    output_dir = project_dir / "frames"
    video_path = project_dir / "three-body-problem.mp4"
    simulation.save_frames(history, output_dir=output_dir, trail_length=50, frame_stride=10)
    simulation.make_video(output_dir, video_path)


if __name__ == "__main__":
    main()
