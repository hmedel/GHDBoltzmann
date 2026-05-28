#!/usr/bin/env julia
"""
P7 figures: reads HDF5 scan results and generates PDF plots.

Usage:
  julia plot_p7_figures.jl <results_dir>
  julia plot_p7_figures.jl results/p7_scan_b_20260520_...

Generates:
  fig_p7_R2_channelA.pdf  — R₂(τ_r) for each ε (inelasticity only)
  fig_p7_R2_channelB.pdf  — R₂(τ_r) for each α (OVM only)
  fig_p7_R2_combined.pdf  — R₂(τ_r) heatmap over (ε, α)
  fig_p7_modes.pdf        — |ρ̂_n|(t) time traces for selected params
"""

using HDF5, CairoMakie, LaTeXStrings, Printf

const FIG_DIR = joinpath(@__DIR__, "figures")
mkpath(FIG_DIR)

# ─── Data loading ────────────────────────────────────────────────────────────

struct ScanPoint
    ε::Float64
    α::Float64
    τ_r::Float64
    R2_peak::Float64
    R2_win::Float64
    R2_win_std::Float64
    rho_n::Matrix{Float64}
    times::Vector{Float64}
    t_peak_idx::Int
end

function load_scan_dir(dir)
    points = ScanPoint[]
    for f in readdir(dir; join=true)
        endswith(f, ".h5") || continue
        h5open(f, "r") do fh
            has_win = haskey(fh, "R_2_win_mean")
            R2p = has_win ? read(fh, "R_2_peak_mean") : read(fh, "R_2_mean")
            R2w = has_win ? read(fh, "R_2_win_mean")  : R2p
            R2s = has_win ? read(fh, "R_2_win_std")   : read(fh, "R_2_std")
            push!(points, ScanPoint(
                read(fh, "epsilon"),
                read(fh, "alpha"),
                read(fh, "tau_r"),
                R2p, R2w, R2s,
                read(fh, "rho_n"),
                read(fh, "times"),
                read(fh, "t_peak_idx"),
            ))
        end
    end
    sort!(points, by=p -> (p.ε, p.α, p.τ_r))
    return points
end

function group_by(points, field)
    groups = Dict{Float64, Vector{ScanPoint}}()
    for p in points
        key = getfield(p, field)
        push!(get!(groups, key, ScanPoint[]), p)
    end
    return groups
end

# ─── Colors & markers ───────────────────────────────────────────────────────

const PALETTE = [:steelblue, :darkorange, :forestgreen, :firebrick, :purple, :teal]
const MARKERS = [:circle, :utriangle, :diamond, :rect, :pentagon, :star5]

# ─── Figure: R₂(τ_r) curves ─────────────────────────────────────────────────

function fig_R2_curves(points, group_field, param_label, filename;
                       ylims=(-0.02, 1.2), ref_R2=nothing, use_win=true)
    groups = group_by(points, group_field)
    keys_sorted = sort(collect(keys(groups)))

    fig = Figure(size=(520, 380), fontsize=12)
    ylabel = use_win ? L"R_2^{\mathrm{win}} \equiv |\hat\rho_2| / |\hat\rho_1|" :
                       L"R_2 \equiv |\hat\rho_2| / |\hat\rho_1|"
    ax = Axis(fig[1, 1],
        xlabel=L"\tau_r",
        ylabel=ylabel,
        limits=(nothing, ylims))

    hlines!(ax, [0.5], color=:gray70, linestyle=:dash, linewidth=0.8,
            label=L"R_2 = 0.5")

    if ref_R2 !== nothing
        hlines!(ax, [ref_R2], color=:gray40, linestyle=:dot, linewidth=0.8,
                label=latexstring("\\mathrm{baseline}\\;R_2 \\approx $(round(ref_R2, digits=2))"))
    end

    for (i, k) in enumerate(keys_sorted)
        pts = groups[k]
        τs = [p.τ_r for p in pts]
        R2 = use_win ? [p.R2_win for p in pts] : [p.R2_peak for p in pts]
        σ2 = [p.R2_win_std for p in pts]
        ci = mod1(i, length(PALETTE))

        errorbars!(ax, τs, R2, σ2,
                   color=(PALETTE[ci], 0.4), whiskerwidth=5)
        scatter!(ax, τs, R2, markersize=8,
                 color=PALETTE[ci], marker=MARKERS[ci],
                 label=latexstring("$(param_label) = $(k)"))
        lines!(ax, τs, R2, color=PALETTE[ci], linewidth=1.5)
    end

    axislegend(ax, position=:rt, framevisible=false, labelsize=10)

    save(joinpath(FIG_DIR, filename), fig)
    println("Saved $(filename)")
    return fig
end

# ─── Figure: mode time traces ───────────────────────────────────────────────

function fig_mode_traces(points; n_panels=4)
    sel = points[round.(Int, range(1, length(points), length=min(n_panels, length(points))))]

    fig = Figure(size=(700, 200 * length(sel)), fontsize=11)

    for (row, p) in enumerate(sel)
        ax = Axis(fig[row, 1],
            xlabel= row == length(sel) ? L"t" : "",
            ylabel=L"|\hat\rho_n|",
            title=latexstring(@sprintf("\\varepsilon=%.2f,\\;\\alpha=%.2f,\\;\\tau_r=%.1f",
                                       p.ε, p.α, p.τ_r)))

        n_modes = size(p.rho_n, 1)
        for n in 1:min(n_modes, 4)
            lines!(ax, p.times, p.rho_n[n, :],
                   color=PALETTE[n], linewidth=1.2,
                   label=latexstring("n=$n"))
        end

        vlines!(ax, [p.times[p.t_peak_idx]], color=:black,
                linestyle=:dot, linewidth=0.8)

        if row == 1
            axislegend(ax, position=:rt, framevisible=false, labelsize=9)
        end
    end

    save(joinpath(FIG_DIR, "fig_p7_modes.pdf"), fig)
    println("Saved fig_p7_modes.pdf")
    return fig
end

# ─── Figure: R₂ heatmap over (param, τ_r) ───────────────────────────────────

function fig_R2_heatmap(points, group_field, param_label, filename; use_win=true)
    groups = group_by(points, group_field)
    keys_sorted = sort(collect(keys(groups)))

    τ_vals = sort(unique([p.τ_r for p in points]))
    R2_mat = fill(NaN, length(keys_sorted), length(τ_vals))

    for (i, k) in enumerate(keys_sorted)
        for p in groups[k]
            j = findfirst(==(p.τ_r), τ_vals)
            j !== nothing && (R2_mat[i, j] = use_win ? p.R2_win : p.R2_peak)
        end
    end

    fig = Figure(size=(560, 340), fontsize=12)
    ax = Axis(fig[1, 1],
        xlabel=L"\tau_r",
        ylabel=param_label,
        yticks=(1:length(keys_sorted),
                [latexstring(@sprintf("%.2f", k)) for k in keys_sorted]))

    hm = heatmap!(ax, 1:length(τ_vals), 1:length(keys_sorted), R2_mat',
                  colormap=:inferno, colorrange=(0, max(1.0, maximum(filter(!isnan, R2_mat)))))
    ax.xticks = (1:length(τ_vals),
                 [latexstring(@sprintf("%.1f", τ)) for τ in τ_vals])

    Colorbar(fig[1, 2], hm, label=L"R_2")

    save(joinpath(FIG_DIR, filename), fig)
    println("Saved $(filename)")
    return fig
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main()
    if length(ARGS) < 1
        # Auto-detect most recent results directory
        res_base = joinpath(@__DIR__, "results")
        if !isdir(res_base)
            error("No results/ directory. Run run_p7_scan.jl first, or pass a results path.")
        end
        dirs = filter(d -> startswith(d, "p7_scan_"), readdir(res_base))
        if isempty(dirs)
            error("No p7_scan_* directories in results/")
        end
        dir = joinpath(res_base, last(sort(dirs)))
        println("Auto-detected: $dir")
    else
        dir = ARGS[1]
    end

    points = load_scan_dir(dir)
    println("Loaded $(length(points)) scan points from $dir")

    εs = unique(p.ε for p in points)
    αs = unique(p.α for p in points)

    has_eps_variation = length(εs) > 1 && all(α == 0.0 for α in αs)
    has_alpha_variation = length(αs) > 1 && all(ε == 0.0 for ε in εs)
    has_both = length(εs) > 1 && length(αs) > 1

    if has_eps_variation || (length(εs) > 1 && !has_both)
        fig_R2_curves(points, :ε, "\\varepsilon", "fig_p7_R2_channelA.pdf";
                      ylims=(-0.02, 1.05), ref_R2=0.16)
        fig_R2_heatmap(points, :ε, L"\varepsilon", "fig_p7_heatmap_channelA.pdf")
    end

    if has_alpha_variation || (length(αs) > 1 && !has_both)
        fig_R2_curves(points, :α, "\\alpha", "fig_p7_R2_channelB.pdf";
                      ref_R2=0.16)
        fig_R2_heatmap(points, :α, L"\alpha", "fig_p7_heatmap_channelB.pdf")
    end

    if has_both
        fig_R2_heatmap(points, :α, L"\alpha", "fig_p7_R2_combined.pdf")
    end

    fig_mode_traces(points)

    println("\nDone. Figures in $(FIG_DIR)")
end

main()
