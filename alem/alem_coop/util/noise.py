from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from jaxtyping import Array, Float


def interpolant(t):
    """Apply the quintic smoothing curve used by Perlin interpolation.

    Args:
        t: Fractional grid coordinates.

    Returns:
        Smoothed interpolation weights.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
    rng: jax.Array,
    shape: tuple[int, int],
    res: tuple[int, int],
    interpolant=interpolant,
    override_angles: Float[Array, "H W"] | None = None,
) -> Float[Array, "H W"]:
    """Generate a two-dimensional tileable Perlin noise field.

    Args:
        rng: JAX random key used to sample gradient directions.
        shape: Height and width of the output field.
        res: Number of noise periods along each axis.
        interpolant: Smoothing function applied to fractional coordinates.
        override_angles: Optional deterministic gradient-angle samples.

    Returns:
        Perlin noise values with the requested shape.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = jnp.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    rng, _rng = jax.random.split(rng)
    if override_angles is not None:
        angles = 2 * jnp.pi * override_angles
    else:
        angles = 2 * jnp.pi * jax.random.uniform(_rng, (res[0] + 1, res[1] + 1))
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    # Ramps
    n00 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return jnp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
    rng: jax.Array,
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: int = 2,
    interpolant=interpolant,
    override_angles: Float[Array, "H W"] | None = None,
) -> Float[Array, "H W"]:
    """Combine normalized Perlin octaves into fractal noise.

    Args:
        rng: JAX random key used to sample each octave.
        shape: Height and width of the output field.
        res: Base number of noise periods along each axis.
        octaves: Number of frequency layers to combine.
        persistence: Amplitude multiplier between octaves.
        lacunarity: Frequency multiplier between octaves.
        interpolant: Smoothing function passed to Perlin generation.
        override_angles: Optional deterministic gradient-angle samples.

    Returns:
        Noise field normalized to the unit interval.
    """
    noise = jnp.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        rng, _rng = jax.random.split(rng)
        noise += amplitude * generate_perlin_noise_2d(
            _rng,
            shape,
            (frequency * res[0], frequency * res[1]),
            interpolant,
            override_angles=override_angles,
        )
        frequency *= lacunarity
        amplitude *= persistence

    # Normalise — guard against constant noise (max == min) to avoid NaN
    noise_range = noise.max() - noise.min()
    noise = (noise - noise.min()) / jnp.where(noise_range > 0, noise_range, 1.0)

    return noise


def main():
    """Render an example fractal-noise field for manual inspection."""
    rng = jax.random.PRNGKey(0)
    noise = generate_fractal_noise_2d(rng, (256, 256), (16, 16), octaves=4)
    plt.imshow(noise, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
