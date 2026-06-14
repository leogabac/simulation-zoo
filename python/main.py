from dataclasses import dataclass
import numpy as np

_UNSET = object()

@dataclass
class PeriodicSquareBox2D:
    """
    2D square simulation box with periodic boundary conditions.
    """

    box_length: float

    def __post_init__(self):
        self.half_box_length = self.box_length / 2

    def wrap(self, coords: np.array) -> np.array:
        """
        Wrap coordinates into [0, L).
        """
        return np.mod(coords, self.box_length)

    def displacement(self, coords1: np.array, coords2: np.array) -> np.array:
        """
        Minimum-image displacement vector r2 - r1.
        """
        dr = np.asarray(coords2) - np.asarray(coords1)
        dr = dr - self.box_length * np.round(dr / self.box_length)
        return dr

    def distance(
        self, coords1: np.array, coords2: np.array, with_disp: bool = False
    ) -> np.array:
        """
        Minimum-image distance.
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
    Basic particle dataclass
    """

    def __init__(self, mass: float, time: float, coords: list, vel: list | object):

        self.time = time
        self.mass = mass
        self.coords = np.asarray(coords, dtype=float)
        self.vel = _UNSET if vel is _UNSET else np.asarray(vel, dtype=float)

    def x(self):
        return self.coords[0]

    def y(self):
        return self.coords[1]

    def vx(self):
        if self.vel is _UNSET:
            return _UNSET
        return self.vel[0]

    def vy(self):
        if self.vel is _UNSET:
            return _UNSET
        return self.vel[1]

    def distance_to(self, coords2: np.array, box: PeriodicSquareBox2D):
        return box.distance(self.coords, coords2)

    def __repr__(self):
        vel = "<unset>" if self.vel is _UNSET else self.vel
        return f"Particle with m={self.mass} @ t={self.time}, x={self.coords}, v={vel}"


class ParticlesInaBox(list):
    def __init__(
        self, particles: list[Particle] = None, box: PeriodicSquareBox2D = None
    ):
        self.particles = list(particles) if particles is not None else []
        self.box = box

    def print_particles(self):
        for p in self.particles:
            print(p)

    def __repr__(self):
        return f"System with {len(self.particles)} particles in box of size {self.box.box_length}"


class GravitationalInteraction:
    def __init__(self, grav_ct: float = 1):
        self.grav_ct = grav_ct

    def energy_scale(self, p1, p2):
        return -self.grav_ct * p1.mass * p2.mass

    def energy(self, p1: Particle, p2: Particle, box: PeriodicSquareBox2D):
        return self.energy_scale(p1, p2) / box.distance(p1.coords, p2.coords)

    def force(self, p1: Particle, p2: Particle, box: PeriodicSquareBox2D):
        """
        Force acting on particle 1 by particle 2
        """

        # here the order of the input changes bc i need to compute the displacement vector
        scale = self.energy_scale(p2, p1)
        distance, dr = box.distance(p2.coords, p1.coords, with_disp=True)
        return (scale / distance**3) * dr


class ThreeBodySimulation:
    def __init__(
        self,
        system: ParticlesInaBox,
        interaction: GravitationalInteraction,
        time_step: float,
    ):
        self.system = system
        self.interaction = interaction
        self.time_step = time_step

    def compute_forces(self, particles: list[Particle]) -> list[np.array]:
        forces = [np.zeros(2, dtype=float) for _ in particles]

        for i, p1 in enumerate(particles):
            for j in range(i + 1, len(particles)):
                p2 = particles[j]
                fij = self.interaction.force(p1, p2, self.system.box)
                forces[i] += fij
                forces[j] -= fij

        return forces

    def compute_accelerations(self, particles: list[Particle]) -> list[np.array]:
        forces = self.compute_forces(particles)
        return [force / particle.mass for force, particle in zip(forces, particles)]

    def copy_particles(self, particles: list[Particle]) -> list[Particle]:
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
        accelerations = self.compute_accelerations(self.system.particles)
        evolved_particles = []

        for previous_particle, current_particle, acceleration in zip(
            previous_particles, self.system.particles, accelerations
        ):
            next_coords = (
                2 * current_particle.coords
                - previous_particle.coords
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

    def run(self, n_steps: int):
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative.")

        history = [self.copy_particles(self.system.particles)]

        if n_steps == 0:
            return history

        previous_particles = self.initialize_previous_particles()

        for _ in range(n_steps):
            next_particles = self.evolve(previous_particles)
            previous_particles = self.system.particles
            self.system.particles = next_particles
            history.append(self.copy_particles(next_particles))

        return history


def main():

    p1 = Particle(mass=1, time=0, coords=[0, 0], vel=[0, 0])
    p2 = Particle(mass=1, time=0, coords=[1, 1], vel=[0, 0])
    p3 = Particle(mass=1, time=0, coords=[-1, -1], vel=[0, 0])
    box = PeriodicSquareBox2D(box_length=20)
    system = ParticlesInaBox([p1, p2, p3], box=box)

    interaction = GravitationalInteraction()
    simulation = ThreeBodySimulation(system, interaction, time_step=0.01)
    history = simulation.run(n_steps=10)

    for step, particles in enumerate(history):
        print(f"step={step}")
        for particle in particles:
            print(particle)


if __name__ == "__main__":
    main()
