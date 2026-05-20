#!/usr/bin/env julia
"""Post-process HDF5 mode data: full time evolution of |ρ̂_n(t)| and R_2(t)."""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "Papers", "Paper01"))
using HDF5, Printf, CairoMakie

const RESULT_DIR = ARGS[1]
const TAU_GRID = [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
const COLORS = [:black, :blue, :green, :orange, :red, :purple, :brown]

function load_run(dir, τ_r)
    fname = joinpath(dir, @sprintf("modes_tau%.2f.h5", τ_r))
    h5open(fname, "r") do fh
        fc = read(fh, "f_cos_mean")
        fs = read(fh, "f_sin_mean")
        t  = read(fh, "times")
        rho_n = sqrt.(fc.^2 .+ fs.^2)
        return t, rho_n
    end
end

function main()
    println("Loading from: ", RESULT_DIR)

    # ── Fig 1: |ρ̂_n(t)| for each τ_r (7 panels) ──
    fig1 = Figure(size=(1400, 1200))
    for (idx, τ_r) in enumerate(TAU_GRID)
        row = div(idx-1, 4) + 1
        col = mod1(idx, 4)
        ax = Axis(fig1[row, col],
                  title = @sprintf("τ_r = %.1f", τ_r),
                  xlabel = row == 2 ? "t" : "",
                  ylabel = col == 1 ? "|ρ̂_n|" : "",
                  yscale = log10,
                  limits = (nothing, (1e-4, 1.0)))
        t, rho_n = load_run(RESULT_DIR, τ_r)
        n_modes = size(rho_n, 1)
        mode_colors = [:red, :blue, :green, :orange, :purple, :cyan, :brown]
        for n in 1:min(n_modes, 5)
            lines!(ax, t, max.(rho_n[n, :], 1e-5),
                   label = "n=$n", color = mode_colors[n], linewidth = n==1 ? 2 : 1)
        end
        if idx == 1
            axislegend(ax, position = :rt, labelsize = 10)
        end
    end
    save(joinpath(RESULT_DIR, "fig_modes_vs_t.png"), fig1, px_per_unit=2)
    println("  Saved fig_modes_vs_t.png")

    # ── Fig 2: R_2(t) = |ρ̂_2(t)|/|ρ̂_1(t)| for all τ_r overlaid ──
    fig2 = Figure(size=(900, 500))
    ax2 = Axis(fig2[1,1],
               xlabel = "t", ylabel = "R₂ = |ρ̂₂|/|ρ̂₁|",
               title = "Even-mode ratio R₂(t)")
    for (idx, τ_r) in enumerate(TAU_GRID)
        t, rho_n = load_run(RESULT_DIR, τ_r)
        R2 = rho_n[2, :] ./ max.(rho_n[1, :], 1e-10)
        R2_clamp = min.(R2, 5.0)
        lines!(ax2, t, R2_clamp,
               label = @sprintf("τ_r=%.1f", τ_r),
               color = COLORS[idx], linewidth = 1.5)
    end
    hlines!(ax2, [0.5], color = :gray, linestyle = :dash, linewidth = 1,
            label = "Burgers target")
    hlines!(ax2, [0.043], color = :gray, linestyle = :dot, linewidth = 1,
            label = "P5 floor")
    axislegend(ax2, position = :rt, labelsize = 10)
    save(joinpath(RESULT_DIR, "fig_R2_vs_t.png"), fig2, px_per_unit=2)
    println("  Saved fig_R2_vs_t.png")

    # ── Fig 3: |ρ̂_1(t)| for all τ_r overlaid ──
    fig3 = Figure(size=(900, 500))
    ax3 = Axis(fig3[1,1],
               xlabel = "t", ylabel = "|ρ̂₁|",
               title = "Fundamental mode |ρ̂₁(t)|")
    for (idx, τ_r) in enumerate(TAU_GRID)
        t, rho_n = load_run(RESULT_DIR, τ_r)
        lines!(ax3, t, rho_n[1, :],
               label = @sprintf("τ_r=%.1f", τ_r),
               color = COLORS[idx], linewidth = 1.5)
    end
    axislegend(ax3, position = :rt, labelsize = 10)
    save(joinpath(RESULT_DIR, "fig_rho1_vs_t.png"), fig3, px_per_unit=2)
    println("  Saved fig_rho1_vs_t.png")

    # ── Fig 4: max_t R_2(t) and R_2 at t_peak vs τ_r ──
    fig4 = Figure(size=(700, 450))
    ax4 = Axis(fig4[1,1],
               xlabel = "τ_r", ylabel = "R₂",
               title = "R₂ diagnostics vs τ_r")
    R2_at_peak = Float64[]
    R2_max = Float64[]
    R2_max_late = Float64[]
    for τ_r in TAU_GRID
        t, rho_n = load_run(RESULT_DIR, τ_r)
        R2 = rho_n[2, :] ./ max.(rho_n[1, :], 1e-10)
        tp = argmax(rho_n[1, :])
        push!(R2_at_peak, R2[tp])
        push!(R2_max, maximum(R2[2:end]))
        # max R2 for t > 10 (after initial transient)
        i10 = findfirst(t .>= 10.0)
        push!(R2_max_late, i10 === nothing ? NaN : maximum(R2[i10:end]))
    end
    scatter!(ax4, TAU_GRID, R2_at_peak, label = "R₂ at t_peak", color = :blue, markersize = 10)
    scatter!(ax4, TAU_GRID, R2_max, label = "max_t R₂(t)", color = :red, markersize = 10)
    scatter!(ax4, TAU_GRID, R2_max_late, label = "max R₂(t>10)", color = :green, markersize = 10)
    hlines!(ax4, [0.5], color = :gray, linestyle = :dash, label = "Burgers target")
    hlines!(ax4, [0.043], color = :gray, linestyle = :dot, label = "P5 floor")
    axislegend(ax4, position = :lt, labelsize = 10)
    save(joinpath(RESULT_DIR, "fig_R2_summary.png"), fig4, px_per_unit=2)
    println("  Saved fig_R2_summary.png")

    # ── Print summary table ──
    println("\n─── Summary ───")
    @printf("%-6s  %6s  %8s  %8s  %8s\n",
            "τ_r", "t_peak", "R₂@peak", "maxR₂", "maxR₂>10")
    for (i, τ_r) in enumerate(TAU_GRID)
        t, rho_n = load_run(RESULT_DIR, τ_r)
        tp = argmax(rho_n[1, :])
        @printf("%-6.1f  %6.2f  %8.4f  %8.4f  %8.4f\n",
                τ_r, t[tp], R2_at_peak[i], R2_max[i], R2_max_late[i])
    end
end

main()
