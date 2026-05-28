#!/usr/bin/env julia
# Generate SIAM talk figures: LWR on 4 geometries
# Produces: siam_fundamental_4geom.png, siam_kymographs_4geom.png

using Pkg
Pkg.activate(joinpath(homedir(), "Science/CollectiveDynamics/ShockWaves/Curves3D"))

using ShockWavesOnCurves
using Plots; gr()
using Printf, Statistics, StaticArrays

const v_max = 30.0
const ρ_max = 1.0
const α = 1.0
const β_κ = 1.0
const β_τ = 0.5
const ρ_L = 0.2
const ρ_R = 0.6
const Ns = 500
const T_final = 10.0

OUTDIR = joinpath(homedir(), "Science/CollectiveDynamics/ShockWaves/Curves3D/siam_figs")
mkpath(OUTDIR)

# ── Define 4 geometries ──────────────────────────────────────────────────

line = StraightLine(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0))
circle = Circle(SVector(0.0, 0.0, 0.0), 50.0, SVector(0.0, 0.0, 1.0))
helix = Helix(50.0, 100.0)
modulated = GeneralizedModulatedCircle(10.0, 10.0, 2)

# :arc = simulate_continuum (arc-length), :ang = simulate_angular_optimized
geoms = [
    ("Straight", line, 100.0, :absorbing, :arc),
    ("Circle",   circle, 2π*50.0, :periodic, :arc),
    ("Helix",    helix, nothing, :periodic, :arc),
    ("Modulated", modulated, nothing, :periodic, :ang),
]

# ── Fig 1: Fundamental diagrams ──────────────────────────────────────────

println("Generating fundamental diagrams...")
ρ_range = range(0, ρ_max, length=200)
colors = [:black, :steelblue, :forestgreen, :firebrick]

p_fd = plot(size=(700, 450), xlabel="ρ", ylabel="Flux  f(ρ) = ρ v(ρ; κ, τ)",
            title="Fundamental Diagram — LWR on Four Geometries",
            legend=:topright, framestyle=:box, dpi=200)

for (i, (name, curve, L, _, _)) in enumerate(geoms)
    if curve isa StraightLine
        κ_rep, τ_rep = 0.0, 0.0
    elseif curve isa Circle
        κ_rep = curvature(curve, 0.0)
        τ_rep = 0.0
    elseif curve isa Helix
        κ_rep = curvature(curve, 0.0)
        τ_rep = torsion(curve, 0.0)
    else
        κ_rep = curvature(curve, 0.0)
        τ_rep = torsion(curve, 0.0)
    end
    g = exp(-β_κ * abs(κ_rep) - β_τ * abs(τ_rep))
    flux = [ρ * v_max * g * (1 - ρ / ρ_max)^α for ρ in ρ_range]
    lab = @sprintf("%s (g=%.4f)", name, g)
    plot!(p_fd, ρ_range, flux, lw=2.5, color=colors[i], label=lab)
end

savefig(p_fd, joinpath(OUTDIR, "siam_fundamental_4geom.png"))
println("  → siam_fundamental_4geom.png")

# ── Run simulations ──────────────────────────────────────────────────────

results = Dict{String, Any}()

for (name, curve, L_override, bc, mode) in geoms
    println("Simulating $name...")

    if mode == :arc
        if curve isa Helix
            L = 2π * sqrt(curve.radius^2 + (curve.pitch/(2π))^2)
        elseif L_override !== nothing
            L = L_override
        else
            L = 100.0
        end
        s_grid = range(0.0, L, length=Ns) |> collect
        ρ_init = [s < L / 2 ? ρ_L : ρ_R for s in s_grid]
        bk = curve isa StraightLine ? 0.0 : β_κ
        bt = curve isa StraightLine ? 0.0 : β_τ
        model = ContinuumModel(v_max, ρ_max, α, bk, bt, curve)
        history, times = simulate_continuum(model, ρ_init, s_grid, T_final,
                                            boundary=bc, save_every=0.1)
        results[name] = (x=s_grid, history=history, times=times, xlab="s")
    else
        Nφ = Ns
        φ_grid = collect(range(0, 2π, length=Nφ + 1)[1:Nφ])
        geom_cache = GeometryCache(curve, φ_grid)

        ρ_φ_init = zeros(Nφ)
        for j in 1:Nφ
            ρ_s = φ_grid[j] < π ? ρ_L : ρ_R
            ρ_φ_init[j] = ρ_s * geom_cache.h[j]
        end

        model = ContinuumModel(v_max, ρ_max, α, β_κ, β_τ, curve)
        history, times = simulate_angular_optimized(model, ρ_φ_init,
                                                    geom_cache, T_final,
                                                    cfl=0.4, order=2,
                                                    limiter=:minmod,
                                                    save_every=0.1)

        ρ_phys = [h[:] ./ geom_cache.h for h in history]
        results[name] = (x=φ_grid, history=ρ_phys, times=times, xlab="φ",
                         geom=geom_cache)
    end
    println("  ✓ $(length(results[name].times)) snapshots")
end

# ── Fig 2: Kymographs (2×2) ─────────────────────────────────────────────

println("Generating kymographs...")
p_ky = plot(layout=(2, 2), size=(900, 650), dpi=200)

for (idx, (name, _, _, _, _)) in enumerate(geoms)
    r = results[name]
    nt = length(r.times)
    nx = length(r.x)

    Z = zeros(nt, nx)
    for ti in 1:nt
        Z[ti, :] = r.history[ti]
    end

    heatmap!(p_ky, r.x, r.times, Z, subplot=idx,
             xlabel=r.xlab, ylabel="t", title=name,
             colorbar=true, c=:viridis, clims=(0.0, 1.0))
end

savefig(p_ky, joinpath(OUTDIR, "siam_kymographs_4geom.png"))
println("  → siam_kymographs_4geom.png")

# ── Fig 3: Fundamental diagram with variable κ (modulated) ──────────────

println("Generating modulated fundamental diagram envelope...")
r_mod = results["Modulated"]
κ_vals = [curvature(modulated, φ) for φ in r_mod.x]
κ_min, κ_max_val = extrema(abs.(κ_vals))

p_env = plot(size=(700, 450), xlabel="ρ", ylabel="Flux f(ρ)",
             title="Fundamental Diagram — Modulated Curve (κ varies 800×)",
             legend=:topright, framestyle=:box, dpi=200)

g_lo = exp(-β_κ * κ_max_val)
g_hi = exp(-β_κ * κ_min)
flux_lo = [ρ * v_max * g_lo * (1 - ρ / ρ_max) for ρ in ρ_range]
flux_hi = [ρ * v_max * g_hi * (1 - ρ / ρ_max) for ρ in ρ_range]
flux_st = [ρ * v_max * (1 - ρ / ρ_max) for ρ in ρ_range]

plot!(p_env, ρ_range, flux_st, lw=1, ls=:dash, color=:gray60,
      label="Straight (reference)")
plot!(p_env, ρ_range, flux_hi, lw=2, color=:steelblue,
      label=@sprintf("Min κ = %.2f (g=%.4f)", κ_min, g_hi))
plot!(p_env, ρ_range, flux_lo, lw=2, color=:firebrick,
      label=@sprintf("Max κ = %.1f (g=%.1e)", κ_max_val, g_lo))

savefig(p_env, joinpath(OUTDIR, "siam_fundamental_envelope.png"))
println("  → siam_fundamental_envelope.png")

println("\nDone. All figures in $OUTDIR")
