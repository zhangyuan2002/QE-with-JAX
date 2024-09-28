#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:14:57 2024

@author: yuan
"""

import jax.numpy as jnp
import jax
import quantecon as qe
import matplotlib.pyplot as plt
from collections import namedtuple

jax.config.update("jax_enable_x64", True)

Parameters = namedtuple("Parameters", 
    ("β",             # discount factor        
     "θ",             # labor productivity
     "c",             # fixed cost in production
     "c_e",           # entry cost
     "w",             # wages
     "m_a",           # productivity shock location parameter
     "σ_a",           # productivity shock scale parameter
     "m_e",           # new entrant location parameter
     "σ_e"))          # new entrant scale parameter

Grids = namedtuple("Grids",
    ("φ_grid",        # productivity grid
     "E_draws",       # entry size draws for Monte Carlo
     "A_draws"))      # productivity shock draws for Monte Carlo

Model = namedtuple("Model",
    ("parameters",    # instance of Parameters
     "grids"))        # instance of Grids

def create_model(β=0.95,             # discount factor
                 θ=0.3,              # labor productivity
                 c=4.0,              # fixed cost in production
                 c_e=1.0,            # entry cost
                 w=1.0,              # wages
                 m_a=-0.012,         # productivity shock location parameter
                 σ_a=0.1,            # productivity shock scale parameter
                 m_e=1.0,            # new entrant location parameter
                 σ_e=0.2,            # new entrant scale parameter
                 φ_grid_max=5,       # productivity grid max
                 φ_grid_size=100,    # productivity grid size
                 E_draw_size=200,    # entry MC integration size
                 A_draw_size=200,    # prod shock MC integration size
                 seed=1234):         # Seed for MC draws
    """
    Create an instance of the `namedtuple` Model using default parameter values.
    """
    
    # Test stability
    assert m_a + σ_a**2 / (2 * (1 - θ)) < 0, "Stability condition fails"
    # Build grids and initialize random number generator
    φ_grid = jnp.linspace(0, φ_grid_max, φ_grid_size)
    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    # Generate a sample of draws of A for Monte Carlo integration
    A_draws = jnp.exp(m_a + σ_a * jax.random.normal(key, (A_draw_size,)))
    # Generate a sample of draws from γ for Monte Carlo
    E_draws = jnp.exp(m_e + σ_e * jax.random.normal(subkey, (E_draw_size,)))
    # Build namedtuple and return
    parameters = Parameters(β, θ, c, c_e, w, m_a, σ_a, m_e, σ_e)
    grids = Grids(φ_grid, E_draws, A_draws)
    model = Model(parameters, grids)
    return model

@jax.jit
def π(φ, p, parameters):
    """ Profits. """
    # Unpack
    β, θ, c, c_e, w, m_a, σ_a, m_e, σ_e = parameters
    # Compute profits
    return (1 - θ) * (p * φ)**(1/(1 - θ)) * (θ/w)**(θ/(1 - θ)) - c 

@jax.jit
def q(φ, p, parameters):
    """ Output. """
    # Unpack
    β, θ, c, c_e, w, m_a, σ_a, m_e, σ_e = parameters
    # Compute output
    return φ**(1/(1 - θ)) * (p * θ/w)**(θ/(1 - θ)) 

def update_cross_section(φ_bar, φ_vec, key, parameters, num_firms):
    # Unpack
    β, θ, c, c_e, w, m_a, σ_a, m_e, σ_e = parameters
    # Update
    Z = jax.random.normal(key, (2, num_firms))  # long rows for row-major arrays
    incumbent_draws = φ_vec * jnp.exp(m_a + σ_a * Z[0, :])
    new_firm_draws = jnp.exp(m_e + σ_e * Z[1, :])
    return jnp.where(φ_vec >= φ_bar, incumbent_draws, new_firm_draws)

update_cross_section = jax.jit(update_cross_section, static_argnums=(4,))

def simulate_firms(φ_bar, parameters, grids, 
                   sim_length=200, num_firms=1_000_000, seed=12):
    """
    Simulate a cross-section of firms when the exit threshold is φ_bar.

    """
    # Set initial conditions to the threshold value
    φ_vec = jnp.ones((num_firms,)) * φ_bar 
    key = jax.random.PRNGKey(seed)
    # Iterate forward in time
    for t in range(sim_length):
        key, subkey = jax.random.split(key)
        φ_vec = update_cross_section(φ_bar, φ_vec, subkey, parameters, num_firms)
    return φ_vec

@jax.jit
def _compute_exp_value_at_phi(v, φ, grids):
    """
    Compute 
    
        E[v(φ')| φ] = Ev(A φ) 
        
    using linear interpolation and Monte Carlo. 
    """
    # Unpack
    φ_grid, E_draws, A_draws = grids
    # Set up V
    Aφ = A_draws * φ 
    vAφ  = jnp.interp(Aφ, φ_grid, v)  # v(A_j φ) for all j
    # Return mean 
    return jnp.mean(vAφ)     # (1/n) Σ_j v(A_j φ)

compute_exp_value_at_phi = jax.vmap(_compute_exp_value_at_phi, (None, 0, None))

@jax.jit
def compute_exp_value(v, grids):
    """
    Compute 
    
        E[v(φ_prime)| φ] = Ev(A φ) for all φ, as a vector

    """
    # Unpack
    φ_grid, E_draws, A_draws = grids
    return compute_exp_value_at_phi(v, φ_grid, grids)

@jax.jit
def T(v, p, parameters, grids):
    """ Bellman operator. """
    # Unpack
    β, θ, c, c_e, w, m_a, σ_a, m_e, σ_e = parameters
    φ_grid, E_draws, A_draws = grids
    # Compute Tv
    EvAφ = compute_exp_value(v, grids)
    return π(φ_grid, p, parameters) + β * jnp.maximum(0.0, EvAφ)

@jax.jit
def get_threshold(v, grids):
    """ Compute the exit threshold. """
    # Unpack
    φ_grid, E_draws, A_draws = grids
    # Compute exit threshold: φ such that E v(A φ) = 0
    EvAφ = compute_exp_value(v, grids)
    i = jnp.searchsorted(EvAφ, 0.0)
    return φ_grid[i]

@jax.jit
def vfi(p, v_init, parameters, grids, tol=1e-6, max_iter=10_000):
    """
    Implement value function iteration to solve for the value function.
    """
    # Unpack
    φ_grid, E_draws, A_draws = grids
    # Set up
    def cond_function(state):
        i, v, error = state
        return jnp.logical_and(i < max_iter, error > tol)

    def body_function(state):
        i, v, error = state
        new_v = T(v, p, parameters, grids)
        error = jnp.max(jnp.abs(v - new_v))
        i += 1
        return i, new_v, error

    # Loop till convergence
    init_state = 0, v_init, tol + 1
    state = jax.lax.while_loop(cond_function, body_function, init_state) 
    i, v, error = state
    return v

@jax.jit
def compute_net_entry_value(p, v_init, parameters, grids):
    """
    Returns the net value of entry, which is 
        
         \int v_bar(φ, p) γ(d φ) - c_e

    This is the break-even condition for new entrants.  The argument
    v_init is used as an initial condition when computing v_bar for VFI.
    """
    c_e = parameters.c_e
    φ_grid = grids.φ_grid
    E_draws = grids.E_draws
    v_bar = vfi(p, v_init, parameters, grids)
    v_φ = jnp.interp(E_draws, φ_grid, v_bar)
    Ev_φ = jnp.mean(v_φ)
    return Ev_φ - c_e, v_bar

def compute_p_star(parameters, grids, p_min=1.0, p_max=2.0, tol=10e-5):
    """
    Compute the equilibrium entry p = 2.0
v_init = jnp.zeros_like(grids.φ_grid)            # Initial condition 
%time v_bar = vfi(p, v_init, parameters, grids).block_until_ready()price p^* via bisection.

    Return both p* and the corresponding value function v_bar, which is
    computed as a byproduct.
    
    Implements the bisection root finding algorithm to find p_star

    """
    φ_grid, E_draws, A_draws = grids
    lower, upper = p_min, p_max
    v_bar = jnp.zeros_like(φ_grid)  # Initial condition at first price guess

    while upper - lower > tol:
        mid = 0.5 * (upper + lower)
        entry_val, v_bar = compute_net_entry_value(mid, v_bar, parameters, grids)
        if entry_val > 0:   # Root is between lower and mid
            lower, upper = lower, mid
        else:               # Root is between mid and upper
            lower, upper = mid, upper

    p_star = 0.5 * (upper + lower)
    return p_star, v_bar

def compute_equilibrium_prices_and_quantities(model):
    """
    Compute 

        1. The equilibrium outcomes for p*, v* and φ*, where φ* is the
           equilibrium exit threshold φ_bar(p*).
        1. The scaling factor necessary to convert the stationary probability
           distribution μ into the equilibrium firm distribution μ* = s μ.
        2. The equilibrium mass of entrants M* = μ*{ φ < φ*}

    """
    # Unpack
    parameters, grids = model
    # Compute prices and values
    p_star, v_bar = compute_p_star(parameters, grids)
    # Get φ_star = φ_bar(p_star), the equilibrium exit threshold 
    φ_star = get_threshold(v_bar, grids)
    # Generate an array of draws from μ, the normalized stationary distribution.
    φ_sample = simulate_firms(φ_star, parameters, grids)
    # Compute s to scale μ
    demand = 1 / p_star
    pre_normalized_supply = jnp.mean(q(φ_sample, p_star, parameters))
    s = demand / pre_normalized_supply
    # Compute M* = μ*{ φ < φ_star}
    m_star = s * jnp.mean(φ_sample < φ_star)
    # return computed objects
    return p_star, v_bar, φ_star, φ_sample, s, m_star

model = create_model()
parameters, grids = model

p = 2.0
v_init = jnp.zeros_like(grids.φ_grid)            # Initial condition 
%time v_bar = vfi(p, v_init, parameters, grids).block_until_ready()

%time v_bar = vfi(p, v_init, parameters, grids).block_until_ready()

p_min, p_max, p_size = 1.0, 2.0, 20
p_vec = jnp.linspace(p_min, p_max, p_size)
entry_vals = []
v_bar = jnp.zeros_like(grids.φ_grid)  # Initial condition at first price guess
for i, p in enumerate(p_vec):
    entry_val, v_bar = compute_net_entry_value(p, v_bar, parameters, grids)
    entry_vals.append(entry_val)
fig, ax = plt.subplots()
ax.plot(p_vec, entry_vals, label="net value of entry")
ax.plot(p_vec, jnp.zeros_like(p_vec), 'k', ls='--', label="break even")
ax.legend()
ax.set_xlabel("price")
ax.set_ylabel("value")
plt.show()

%%time
p_star, v_bar, φ_star, φ_sample, s, m_star = \
        compute_equilibrium_prices_and_quantities(model)
        
%%time
p_star, v_bar, φ_star, φ_sample, s, m_star = \
        compute_equilibrium_prices_and_quantities(model)
        
p_star

fig, ax = plt.subplots()
ax.plot(grids.φ_grid, v_bar, label=r'$\varphi \mapsto \bar v(\varphi, p^*)$')
ax.set_xlabel("productivity")
ax.set_ylabel("firm value")
ax.legend()
plt.show()

output_dist = q(φ_sample, p_star, parameters)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(jnp.log(output_dist), bins=100, density=True,
        label="firm size distribution")
ax.set_xlabel("log output")
ax.set_ylabel("frequency")
ax.legend()
plt.show()

def ECCDF(data):
    """
    Return a function that implements the ECCDF given the data.
    """
    def eccdf(x):
        return jnp.mean(data > x)
    return eccdf

eccdf = ECCDF(output_dist)

ϵ = 10e-10
x_grid = jnp.linspace(output_dist.min() + ϵ, output_dist.max() - ϵ, 200)
y = [eccdf(x) for x in x_grid]

fix, ax = plt.subplots()
ax.loglog(x_grid, y, 'o', label="ECCDF")
ax.set_xlabel("productivity")
ax.set_ylabel("counter CDF")
plt.show()