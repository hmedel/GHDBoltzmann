# GHDBoltzmann

Discrete-velocity solver for the two-species generalised hydrodynamic (GHD) Boltzmann equation on a periodic 1D ring, with binary mass disorder.

Companion code for: H. Medel, *Numerical GHD-Boltzmann solver for mass-disordered hard rods on a Riemannian ring: transport coefficients and the integrability-to-diffusion crossover* (Paper V, 2026).

## Requirements

- Julia >= 1.10
- Packages: `LinearAlgebra`, `Printf`, `Statistics`, `DelimitedFiles`
- For figures: `Plots`, `LaTeXStrings` (GR backend)

No external dependencies beyond the Julia standard library for the core solver.

## Solver module

**`GHDBoltzmannSolver.jl`** — Main solver module.

- Strang operator splitting: R(dt/2) . S(dt) . R(dt/2)
- Streaming: WENO5 upwind reconstruction (5th-order in smooth regions)
- Collision: 3-point Lagrange interpolation (Bobylev-Palczewski-Schneider form), preserves all 4 collision invariants to machine precision
- Dressed velocity: v_eff = (v - eta*u_bar)/(1 - eta), clamped at eta_max = 0.99

### Quick start

```julia
push!(LOAD_PATH, @__DIR__)
using GHDBoltzmannSolver

p = GHDBParams(L=12.57, sigma=0.00314, m_L=1.0, m_H=5.0,
               kT=1.0, N_s=80, N_v=128, V_max=6.0, cfl=0.25)
st = GHDBState(p)
# ... set initial condition on st.rho_L, st.rho_H ...
records, mass_err, energy_err = run_simulation!(st, T_final,
    save_every=0.2, callback=my_observable)
```

## Reproducing Paper V results

### Table I: Sound speed (Sec IV)

```bash
julia test_sound_speed.jl
```

Runs the sound-eigenmode IC at m_max in {5, 10, 100}, extracts c_s^DVM via tight-window phase fit. Expected: c_s^DVM / c_s^an = 0.999x.

### Table II: Interdiffusion D_LH (Sec V)

```bash
julia chapman_enskog.jl
```

Builds the linearised collision operator with 3-pt Lagrange kernel, solves the C-E equation for the composition-gradient source. Expected: D_LH ~ 4.0 across m_max in [1.5, 100].

### Table III: Thermal conductivity lambda_T (Sec V)

```bash
julia thermal_conductivity.jl
julia convergence_and_scaling.jl
```

Species-dependent Eucken correction C_alpha = 1/2 + m_alpha/m_bar. Convergence sweep across N_v, V_max, pseudo-inverse threshold. Expected: lambda_T diverges as (m_max - 1)^{-1.5} near the integrable limit.

### Table IV: Dispersion relation (Sec VII)

```bash
julia dispersion_relation.jl
julia hydro_projection.jl
```

Full 2N_v eigenvalue problem M(k) = k*A + i*L. Euler projection recovers c_s^an to 4 decimal places. Kinetic-regime saturation at Kn > 1.

### Tables V-VI: Step-IC mode ratios (Sec VIII)

```bash
julia long_step_sim.jl
julia analyze_step.jl
```

Step-velocity IC at U_0 = 0.05 (linear) and U_0 = 0.5 (nonlinear), 12 sound periods. Odd-mode 1/n ratio and even-mode saturation.

### Figures

```bash
julia make_figures.jl
```

Generates all PDF figures in `../Paper02/figures/`.

### Hyperscaling tests (Sec VI)

```bash
julia hyperscaling_sweep.jl     # eigenvalue-based kappa_eff(k, m)
julia greenkubo.jl              # Green-Kubo lambda_T(k)
julia heat_autocorrelator.jl    # time-resolved J_E(t) relaxation
julia heat_extended.jl          # extended L sweep (35 pts)
julia heat_largeL.jl            # large-L sweep (L=200,300,400)
```

### Diagnostic / audit scripts

```bash
julia test_3pt_kernel.jl        # verify 4 collision invariants
julia test_dn_decoupling.jl     # verify delta-n decoupling
julia audit_DLH.jl              # D_LH convergence audit
julia audit_lambdaT.jl          # lambda_T convergence audit
julia full_dispersion_check.jl  # full 2N_v spectrum at k_1
```

## Parameters

Default parameters matching Refs. [PaperII, PaperIII]:

| Parameter | Value | Description |
|-----------|-------|-------------|
| L | 12.57 | Ring perimeter |
| sigma | 0.00314 | Rod diameter |
| rho_tot | 31.82 | Total number density (N=400 rods) |
| eta | 0.10 | Packing fraction |
| kT | 1.0 | Temperature |
| m_L | 1.0 | Light species mass |
| m_H | 5.0 | Heavy species mass (varied) |

## License

Research code accompanying an academic publication. Please cite the paper if you use this code.
