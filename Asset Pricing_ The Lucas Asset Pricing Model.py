#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 23:20:03 2024

@author: yuan
"""

import jax.numpy as jnp
import jax
import numpy as np
import numba
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from time import time

def create_lucas_tree_model(γ=2,            # CRRA utility parameter
                            β=0.95,         # Discount factor
                            α=0.90,         # Correlation coefficient
                            σ=0.1,          # Volatility coefficient
                            grid_size=500,
                            draw_size=1_000,
                            seed=11):
        # Set the grid interval to contain most of the mass of the
        # stationary distribution of the consumption endowment
        ssd = σ / np.sqrt(1 - α**2)
        grid_min, grid_max = np.exp(-4 * ssd), np.exp(4 * ssd)
        grid = np.linspace(grid_min, grid_max, grid_size)
        # Set up distribution for shocks
        np.random.seed(seed)
        ϕ = lognorm(σ)
        draws = ϕ.rvs(500)
        # And the vector h
        h = np.empty(grid_size)
        for i, y in enumerate(grid):
            h[i] = β * np.mean((y**α * draws)**(1 - γ))
        # Pack and return
        params = γ, β, α, σ
        arrays = grid, draws, h
        return params, arrays
    
    
@numba.jit
def T(params, arrays, f):
    """
    The Lucas pricing operator.
    """
    # Unpack
    γ, β, α, σ = params
    grid, draws, h = arrays
    # Turn f into a function
    Af = lambda x: np.interp(x, grid, f)
    # Compute Tf and return
    Tf = np.empty_like(f)
    # Apply the T operator to f using Monte Carlo integration
    for i in range(len(grid)):
        y = grid[i]
        Tf[i] = h[i] + β * np.mean(Af(y**α * draws))
    return Tf

def solve_model(params, arrays, tol=1e-6, max_iter=500):
    """
    Compute the equilibrium price function.

    """
    # Unpack
    γ, β, α, σ = params
    grid, draws, h = arrays
    # Set up and loop
    i = 0
    f = np.ones_like(grid)  # Initial guess of f
    error = tol + 1
    while error > tol and i < max_iter:
        Tf = T(params, arrays, f)
        error = np.max(np.abs(Tf - f))
        f = Tf
        i += 1
    price = f * grid**γ  # Back out price vector
    return price

params, arrays = create_lucas_tree_model()
γ, β, α, σ = params
grid, draws, h = arrays

# Solve once to compile
start = time()
price_vals = solve_model(params, arrays)
numba_with_compile_time = time() - start
print("Numba compile plus execution time = ", numba_with_compile_time)

# Now time execution without compile time
start = time()
price_vals = solve_model(params, arrays)
numba_without_compile_time = time() - start
print("Numba execution time = ", numba_without_compile_time)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid, price_vals, label='$p*(y)$')
ax.set_xlabel('$y$')
ax.set_ylabel('price')
ax.legend()
plt.show()

def create_lucas_tree_model(γ=2,            # CRRA utility parameter
                            β=0.95,         # Discount factor
                            α=0.90,         # Correlation coefficient
                            σ=0.1,          # Volatility coefficient
                            grid_size=500,
                            draw_size=1_000,
                            seed=11):
        # Set the grid interval to contain most of the mass of the
        # stationary distribution of the consumption endowment
        ssd = σ / jnp.sqrt(1 - α**2)
        grid_min, grid_max = jnp.exp(-4 * ssd), jnp.exp(4 * ssd)
        grid = jnp.linspace(grid_min, grid_max, grid_size)

        # Set up distribution for shocks
        key = jax.random.key(seed)
        draws = jax.random.lognormal(key, σ, shape=(draw_size,))
        grid_reshaped = grid.reshape((grid_size, 1))
        draws_reshaped = draws.reshape((-1, draw_size))
        h = β * jnp.mean((grid_reshaped**α * draws_reshaped) ** (1-γ), axis=1)
        params = γ, β, α, σ
        arrays = grid, draws, h
        return params, arrays
    
@jax.jit 
def compute_expectation(y, α, draws, grid, f):
    return jnp.mean(jnp.interp(y**α * draws, grid, f))

# Vectorize over y
compute_expectation = jax.vmap(compute_expectation,
                               in_axes=(0, None, None, None, None))


@jax.jit
def T(params, arrays, f):
    """
    The Lucas operator

    """
    grid, draws, h = arrays
    γ, β, α, σ = params
    mci = compute_expectation(grid, α, draws, grid, f)
    return h + β * mci


def successive_approx_jax(T,                     # Operator (callable)
                          x_0,                   # Initial condition
                          tol=1e-6,        # Error tolerance
                          max_iter=10_000):      # Max iteration bound
    def body_fun(k_x_err):
        k, x, error = k_x_err
        x_new = T(x)
        error = jnp.max(jnp.abs(x_new - x))
        return k + 1, x_new, error

    def cond_fun(k_x_err):
        k, x, error = k_x_err
        return jnp.logical_and(error > tol, k < max_iter)

    k, x, error = jax.lax.while_loop(cond_fun, body_fun,
                                     (1, x_0, tol + 1))
    return x


successive_approx_jax = \
    jax.jit(successive_approx_jax, static_argnums=(0,))
    
def solve_model(params, arrays, tol=1e-6, max_iter=500):
    """
    Compute the equilibrium price function.

    """
    # Simplify notation
    grid, draws, h = arrays
    γ, β, α, σ = params
    _T = lambda f: T(params, arrays, f)
    f = jnp.ones_like(grid)  # Initial guess of f

    f = successive_approx_jax(_T, f, tol=tol, max_iter=max_iter)

    price = f * grid**γ  # Back out price vector

    return price

params, arrays = create_lucas_tree_model()
grid, draws, h = arrays
γ, β, α, σ = params

# Solve once to compile
start = time()
price_vals = solve_model(params, arrays).block_until_ready()
jax_with_compile_time = time() - start
print("JAX compile plus execution time = ", jax_with_compile_time)


# Now time execution without compile time
start = time()
price_vals = solve_model(params, arrays).block_until_ready()
jax_without_compile_time = time() - start
print("JAX execution time = ", jax_without_compile_time)
print("Speedup factor = ", numba_without_compile_time/jax_without_compile_time)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid, price_vals, label='$p*(y)$')
ax.set_xlabel('$y$')
ax.set_ylabel('price')
ax.legend()
plt.show()