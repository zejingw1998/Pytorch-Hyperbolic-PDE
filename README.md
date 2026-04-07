# PyTorch Hyperbolic PDE

A research-oriented repository for numerical experiments on **hyperbolic partial differential equations** using **PyTorch**.

This project focuses on implementing and comparing classical numerical methods for **conservation laws**, with particular attention to:

- hyperbolic PDEs
- linear advection and Burgers' equation
- finite difference and finite volume ideas
- CFL stability and numerical instability
- numerical diffusion and shock resolution
- operator splitting methods

The repository is built as a personal computational lab for studying how classical schemes behave in practice, and how implementation choices affect stability, accuracy, and qualitative solution structure.

---

## Motivation

I created this repository to turn PDE theory into working numerical experiments.

Instead of only reading about stability conditions, shock formation, or numerical diffusion, I wanted to:

- implement the schemes myself,
- test them under different resolutions and CFL numbers,
- compare their behavior quantitatively and visually,
- and build reusable code for future work in numerical analysis and scientific computing.

This repository is therefore both a **learning project** and a **research-style coding portfolio**.

---

## Current Topics

The current repository includes work on:

- **Linear advection**
  - transport behavior
  - periodic boundary conditions
  - stability and instability tests
  - comparison of finite-difference schemes

- **Burgers' equation**
  - nonlinear transport
  - shock formation
  - comparison of classical schemes
  - numerical dissipation and total variation behavior

- **Numerical stability analysis**
  - CFL-based time stepping
  - stable vs unstable discretizations
  - behavior under mesh refinement

- **Operator splitting methods**
  - splitting-based numerical ideas
  - experimentation with different update structures

---

## Implemented / Explored Methods

Some of the numerical methods explored in this repository include:

- upwind / backward-difference methods
- FTFS-type and related finite-difference experiments
- Lax-type methods
- Rusanov-type fluxes
- CFL-controlled explicit time stepping
- resolution studies and stability comparisons

---

## Example Questions Studied

This repository is built around questions such as:

- How does CFL affect stability in explicit schemes?
- How diffusive is the Lax method compared with Rusanov-type methods?
- How well do classical methods capture transport, steep gradients, and shocks?
- What qualitative differences appear as the mesh is refined?
- How can PyTorch be used as a convenient computational backend for PDE experiments?

---

## Tech Stack

- **Python**
- **PyTorch**
- **NumPy**
- **Matplotlib**
- Jupyter Notebook

---
