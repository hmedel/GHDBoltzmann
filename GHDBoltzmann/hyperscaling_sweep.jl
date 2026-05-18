#=
Sweep (k, m_max) of the full linearised GHD-Boltzmann
dispersion operator M(k) = k·A + i·L̃ and extract the
heat-mode eigenvalue. Test hyperscaling prediction:

  λ_T^eff(k, m-1) ~ k^{-β} F((m-1)·k^{-ζ})

with β = 1/3, ζ = α^{-1}·β ≈ 0.22 (where α ≈ 1.53 from
Sec V.B). The hyperscaling F is recovered if the data
collapse onto a single curve under the rescaling.

Heat mode identification: eigenvalue with Re(ω) ≈ 0 (non-
propagating) and smallest |Im(ω)| above the trivial zero
modes. Defines effective heat diffusivity
  κ_eff(k, m) ≡ -Im(ω_heat)/k^2
which equals λ_T/(ρ c_p) in the hydrodynamic regime.

Output: CSV table + console summary. Generates a figure
(Fig 5) showing the collapse, if the hyperscaling holds.
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")
include("dispersion_relation.jl")  # build_streaming_A
using LinearAlgebra, Printf, DelimitedFiles
using Plots, LaTeXStrings
gr()
default(framestyle=:box, gridstyle=:dot, gridlinewidth=0.4,
        size=(450, 330), titlefontsize=11, labelfontsize=10,
        legendfontsize=8, tickfontsize=8)

println("="^70)
println("Hyperscaling sweep: κ_eff(k, m_max) from heat-mode eigenvalue")
println("="^70)

function find_heat_mode(m_max::Float64, k::Float64;
                        n_each=rho_tot/2, kT=1.0,
                        N_v::Int=N_v, V_max::Float64=V_max, sigma=sigma_val)
    A, fL, fH, v, dv, η_0 = build_streaming_A(m_max; n_each=n_each, kT=kT,
                                               N_v=N_v, V_max=V_max, sigma=sigma)
    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma, 1.0, m_max, v, fL, fH)
    M = k*A + im*L̃
    ω_all = eigvals(M)

    # Heat mode: Re ≈ 0 (non-propagating) and smallest |Im|.
    # Restrict to Re|ω| < c_s_an·k/3 to avoid sound and streaming modes.
    m̄ = 0.5*(1.0 + m_max)
    c_s_an = sqrt(3/m̄)/(1 - η_0)
    re_cutoff = c_s_an*k / 3.0

    # Discard modes with Im ≈ 0 (exact zero modes from streaming continuum
    # at high v) and those with |Re| above the cutoff
    candidates = filter(ω -> abs(real(ω)) < re_cutoff &&
                              abs(imag(ω)) > 1e-8 &&
                              imag(ω) < 0,
                        ω_all)
    if isempty(candidates)
        return nothing
    end
    # Among non-propagating modes, the heat mode is the LEAST DAMPED.
    heat = candidates[argmin(abs.(imag.(candidates)))]
    return heat
end

# ---- Setup ----
const k_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
const m_grid = [1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
results = Dict{Tuple{Float64,Float64}, ComplexF64}()

println("\nSweep (m_max, k) → ω_heat:")
println("m_max\\k\t" * join([@sprintf("%.2f", k) for k in k_grid], "\t"))
for m in m_grid
    print("$m\t")
    for k in k_grid
        ω = find_heat_mode(m, k)
        if ω !== nothing
            results[(m, k)] = ω
            print(@sprintf("%.3e\t", -imag(ω)))
        else
            print("---\t")
        end
    end
    println()
end

# ---- Effective κ = -Im(ω)/k² ----
println("\nκ_eff(k, m) = -Im(ω_heat)/k²:")
println("m_max\\k\t" * join([@sprintf("%.2f", k) for k in k_grid], "\t"))
κ_data = zeros(length(m_grid), length(k_grid))
for (i, m) in enumerate(m_grid)
    print("$m\t")
    for (j, k) in enumerate(k_grid)
        if haskey(results, (m, k))
            κ = -imag(results[(m, k)])/k^2
            κ_data[i, j] = κ
            print(@sprintf("%.2f\t", κ))
        else
            κ_data[i, j] = NaN
            print("---\t")
        end
    end
    println()
end

# ---- Hyperscaling test ----
# Prediction: κ_eff(k, m-1) = k^{-β} F((m-1)·k^{-ζ})
# with β = 1/3 from anomalous transport, and ζ from α·ζ = β
const α_local = 1.53
const β_local = 1/3
const ζ_local = β_local / α_local  # ≈ 0.218

println("\nHyperscaling prediction: α=$(α_local), β=$(β_local), ζ=$(round(ζ_local,digits=3))")
println("Rescaled κ·k^β  vs  (m-1)·k^{-ζ}:")
xs = Float64[]; ys = Float64[]; labels = String[]
for (i, m) in enumerate(m_grid)
    for (j, k) in enumerate(k_grid)
        if !isnan(κ_data[i, j]) && κ_data[i, j] > 0
            push!(xs, (m-1.0)*k^(-ζ_local))
            push!(ys, κ_data[i, j]*k^β_local)
            push!(labels, "m=$m, k=$k")
        end
    end
end

# Plot
p = scatter(xs, ys, xscale=:log10, yscale=:log10,
            xlabel=L"(m_{\max}-1)\,k^{-\zeta}",
            ylabel=L"\kappa_{\rm eff}\,k^{\beta}",
            title=L"Hyperscaling: \beta=1/3,\ \zeta\approx 0.22",
            ms=4, color=:blue, label="data",
            legend=:topleft)
savefig(p, joinpath(@__DIR__, "..", "Paper02", "figures", "fig_hyperscaling.pdf"))
println("\nFigure saved: fig_hyperscaling.pdf")

# Quantify collapse quality: residual variance of log10(κ k^β) at fixed (m-1)k^{-ζ}
# Bin by log10 of x, compute std of log10 y in each bin
log10_xs = log10.(xs); log10_ys = log10.(ys)
sort_perm = sortperm(log10_xs)
log10_xs_sorted = log10_xs[sort_perm]
log10_ys_sorted = log10_ys[sort_perm]

# Simple smoothness measure: variance of nearest-neighbor differences
local_diffs = diff(log10_ys_sorted)
local_x_diffs = diff(log10_xs_sorted)
slopes = local_diffs ./ max.(abs.(local_x_diffs), 1e-3)
@printf("\nCollapse quality (lower = better):\n")
@printf("  Overall range of log10(κ·k^β): [%.2f, %.2f], spread %.2f\n",
        minimum(log10_ys), maximum(log10_ys), maximum(log10_ys) - minimum(log10_ys))
@printf("  Overall range of log10((m-1)k^{-ζ}): [%.2f, %.2f]\n",
        minimum(log10_xs), maximum(log10_xs))

# Test scaling: in collapse, the rescaled κ at FIXED rescaled x should not depend on (k, m) independently
# Group by rescaled x bins and look at spread
println("\nBy-bin variance of log10(κ·k^β) at fixed (m-1)k^{-ζ} ±0.3 dex:")
n_bins = 6
bins_x = range(minimum(log10_xs), maximum(log10_xs), length=n_bins+1)
for b in 1:n_bins
    mask = (log10_xs .≥ bins_x[b]) .& (log10_xs .< bins_x[b+1])
    if count(mask) >= 2
        @printf("  bin [%.2f, %.2f]: n=%d, std(log10 y) = %.3f\n",
                bins_x[b], bins_x[b+1], count(mask),
                count(mask) > 1 ? sqrt(sum((log10_ys[mask] .- sum(log10_ys[mask])/count(mask)).^2)/(count(mask)-1)) : NaN)
    end
end

# Save raw data
open(joinpath(@__DIR__, "hyperscaling_data.csv"), "w") do io
    write(io, "m_max,k,Re_omega,Im_omega,kappa_eff,rescaled_x,rescaled_y\n")
    for (i, m) in enumerate(m_grid), (j, k) in enumerate(k_grid)
        if haskey(results, (m, k))
            ω = results[(m, k)]
            κ = -imag(ω)/k^2
            rx = (m-1.0)*k^(-ζ_local)
            ry = κ*k^β_local
            @printf(io, "%g,%g,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    m, k, real(ω), imag(ω), κ, rx, ry)
        end
    end
end
println("Raw data: hyperscaling_data.csv")
