#!/usr/bin/env julia
"""
P6 figures: numerical data plots for Paper VI.
Generates two PDF figures:
  fig_p6_R2_all.pdf   — Panel (a): Model B null, (b): OVM-HR scan 2
  fig_p6_resonance.pdf — Resonance scan (α=0.10 and α=0.12)
"""

using CairoMakie, LaTeXStrings

const FIG_DIR = joinpath(@__DIR__, "..", "Paper02", "figures")
mkpath(FIG_DIR)

# ─── Data ────────────────────────────────────────────────────────────────────

# Model B (contact-freeze, 200 seeds): real data from xolotl HDF5
const modelB_τr = [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
const modelB_R2 = [0.1584, 0.0333, 0.0178, 0.0451, 0.0212, 0.0228, 0.0305]
const modelB_σ2 = [0.0032, 0.0045, 0.0045, 0.0046, 0.0045, 0.0045, 0.0045]

# OVM-HR scan 2 (30 seeds each)
const scan2_τr = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

const scan2_R2 = Dict(
    0.05 => [0.1616, 0.1659, 0.3813, 0.1661, 0.1643, 0.1661, 0.1661],
    0.08 => [0.2393, 0.2780, 0.2693, 0.4084, 0.3993, 0.1724, 0.4285],
    0.12 => [0.2955, 0.4569, 0.2696, 0.1778, 0.2454, 0.6475, 0.5331],
    0.15 => [0.5432, 0.3769, 0.4824, 0.3897, 0.5641, 0.2614, 0.3734],
)

# Resonance scan (100 seeds, with σ₂)
const res_τr = [0.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 11.0]
const τ_sound = 11.3  # L/c_s with c_s = v_th/(1-η) ≈ 1.11

# α = 0.10
const res_R2_010 = [0.2822, 0.2692, 0.1525, 0.2893, 0.2585, 0.0894,
                    0.2053, 0.1540, 0.1280, 0.1540, 0.1540, 0.1540]
const res_σ2_010 = [0.0318, 0.0345, 0.0342, 0.0354, 0.0365, 0.0370,
                    0.0358, 0.0365, 0.0379, 0.0350, 0.0367, 0.0371]

# α = 0.12
const res_R2_012 = [0.1769, 0.3536, 0.2796, 0.4205, 0.3274, 0.2888,
                    0.3078, 0.4954, 0.1551, 0.3254, 0.5165, 0.3420]
const res_σ2_012 = [0.0353, 0.0366, 0.0388, 0.0398, 0.0401, 0.0406,
                    0.0412, 0.0417, 0.0414, 0.0420, 0.0425, 0.0430]

# ─── Figure 1: R₂ vs τ_r (Model B + OVM scan 2) ────────────────────────────

function fig_R2_all()
    fig = Figure(size=(700, 340), fontsize=11)

    # Panel (a): Model B
    ax1 = Axis(fig[1, 1],
        xlabel=L"\tau_r",
        ylabel=L"R_2 \equiv |\hat\rho_2| / |\hat\rho_1|",
        title="(a) Model B (contact-freeze, 200 seeds)",
        yticks=0:0.1:0.6,
        limits=((-0.3, 4.5), (-0.02, 0.60)))

    hlines!(ax1, [0.5], color=:gray70, linestyle=:dash, linewidth=0.8)
    vlines!(ax1, [2.8], color=:red, linestyle=:dashdot, linewidth=1.0,
            label=L"\tau_r^c \approx 2.8 \; \mathrm{(predicted)}")
    errorbars!(ax1, modelB_τr, modelB_R2, modelB_σ2,
               color=(:navy, 0.4), whiskerwidth=5)
    scatter!(ax1, modelB_τr, modelB_R2, markersize=8, color=:navy)
    lines!(ax1, modelB_τr, modelB_R2, color=:navy, linewidth=1.5)
    axislegend(ax1, position=:rt, framevisible=false, labelsize=9)

    # Panel (b): OVM-HR scan 2
    ax2 = Axis(fig[1, 2],
        xlabel=L"\tau_r",
        title="(b) OVM-HR (30 seeds)",
        yticks=0:0.1:0.7,
        limits=((-0.5, 9.0), (-0.02, 0.72)))

    hlines!(ax2, [0.5], color=:gray70, linestyle=:dash, linewidth=0.8)
    vlines!(ax2, [2.8], color=:red, linestyle=:dashdot, linewidth=1.0)

    colors = [:steelblue, :darkorange, :forestgreen, :firebrick]
    markers = [:circle, :utriangle, :diamond, :rect]
    α_vals = [0.05, 0.08, 0.12, 0.15]

    for (i, α) in enumerate(α_vals)
        scatter!(ax2, scan2_τr, scan2_R2[α], markersize=7,
                 color=colors[i], marker=markers[i],
                 label=latexstring("\\alpha = $(α)"))
        lines!(ax2, scan2_τr, scan2_R2[α], color=colors[i],
               linewidth=1.2, linestyle=α == 0.15 ? :dash : :solid)
    end
    axislegend(ax2, position=:rt, framevisible=false, labelsize=9)

    save(joinpath(FIG_DIR, "fig_p6_R2_all.pdf"), fig)
    println("Saved fig_p6_R2_all.pdf")
    return fig
end

# ─── Figure 2: Resonance scan ────────────────────────────────────────────────

function fig_resonance()
    fig = Figure(size=(500, 340), fontsize=11)

    ax = Axis(fig[1, 1],
        xlabel=L"\tau_r / \tau_{\mathrm{sound}}",
        ylabel=L"R_2",
        title="Resonance test (100 seeds)",
        limits=((-0.05, 0.95), (-0.02, 0.62)))

    τr_norm = res_τr ./ τ_sound

    hlines!(ax, [0.5], color=:gray70, linestyle=:dash, linewidth=0.8)
    vlines!(ax, [0.5], color=:red, linestyle=:dashdot, linewidth=1.0,
            label=L"\tau_r = \tau_{\mathrm{sound}}/2")

    # α = 0.10 with error bars
    errorbars!(ax, τr_norm, res_R2_010, res_σ2_010,
               color=(:steelblue, 0.5), whiskerwidth=4)
    scatter!(ax, τr_norm, res_R2_010, markersize=7, color=:steelblue,
             marker=:circle, label=L"\alpha = 0.10")
    lines!(ax, τr_norm, res_R2_010, color=:steelblue, linewidth=1.0)

    # α = 0.12 with error bars
    errorbars!(ax, τr_norm, res_R2_012, res_σ2_012,
               color=(:forestgreen, 0.5), whiskerwidth=4)
    scatter!(ax, τr_norm, res_R2_012, markersize=7, color=:forestgreen,
             marker=:diamond, label=L"\alpha = 0.12")
    lines!(ax, τr_norm, res_R2_012, color=:forestgreen, linewidth=1.0)

    axislegend(ax, position=:rt, framevisible=false, labelsize=10)

    save(joinpath(FIG_DIR, "fig_p6_resonance.pdf"), fig)
    println("Saved fig_p6_resonance.pdf")
    return fig
end

# ─── Main ────────────────────────────────────────────────────────────────────

fig_R2_all()
fig_resonance()
println("\nDone. Figures in $(FIG_DIR)")
