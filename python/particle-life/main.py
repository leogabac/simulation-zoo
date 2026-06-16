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

type2color = {
    0: "r",
    1: "g",
    2: "b",
}


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
        self.color = type2color[type_]

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
        ax.scatter([self.x()], [self.y()], s=20, color=self.color)

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

    def __repr__(self) -> str:
        return f"System with {len(self.particles)} particles in box of size {self.box.box_length}"


def main():

    number_particles = 100
    box = PeriodicSquareBox2D(box_length=100)
    box_center = np.array([box.half_box_length, box.half_box_length])

    rng = np.random.default_rng()
    random_particles = [
        Particle(
            mass=1.0,
            time=0,
            coords=box.random_coordinate(),
            vel=np.random.rand(2),
            type_=rng.integers(0, high=3),
        )
        for _ in range(number_particles)
    ]
    system = ParticlesInaBox(random_particles, box)
    system.print_particles()
    print(system)


if __name__ == "__main__":
    main()
