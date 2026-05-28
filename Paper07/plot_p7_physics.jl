#!/usr/bin/env julia
"""
P7 physics figures:
  1. Mode spectrum R_n vs n at late time, compared to Burgers 1/n
  2. |ρ̂₁|(t) growth curves for different ε
  3. R₂(ε) at τ_r=0 showing the transition
  4. Combined heatmap from (ε, α) scan
"""

using HDF5, CairoMakie, LaTeXStrings, Printf

const FIG_DIR = joinpath(@__DIR__, "figures")
mkpath(FIG_DIR)

const PALETTE = [:steelblue, :darkorange, :forestgreen, :firebrick, :purple, :teal]
const MARKERS = [:circle, :utriangle, :diamond, :rect, :pentagon, :star5]

# ─── Load data ──────────────────────────────────────────────────────────────

function load_point(path)
    h5open(path, "r") do fh
        return (
            ε = read(fh, "epsilon"),
            α = read(fh, "alpha"),
            τ = read(fh, "tau_r"),
            rho = read(fh, "rho_n"),
            t = read(fh, "times"),
            KE = haskey(fh, "KE_lost") ? read(fh, "KE_lost") : NaN,
            nc = haskey(fh, "n_colls") ? read(fh, "n_colls") : 0,
        )
    end
end

# ─── Fig 1: Mode spectrum at late time ──────────────────────────────────────

function fig_spectrum()
    dir_a = "results/p7_scan_a_20260520_164805"
    dir_bl = "results/p7_scan_baseline_20260520_164748"

    fig = Figure(size=(520, 380), fontsize=12)
    ax = Axis(fig[1, 1],
        xlabel=L"n",
        ylabel=L"R_n \equiv |\hat\rho_n| / |\hat\rho_1|",
        xticks=1:7,
        limits=(0.5, 7.5, -0.05, 1.15))

    ns = collect(1:7)
    burgers = 1.0 ./ ns
    lines!(ax, ns, burgers, color=:black, linestyle=:dash, linewidth=2,
           label=L"\mathrm{Burgers}\;1/n")

    t_eval = 30.0

    for (i, (label, dir, file, col)) in enumerate([
        (L"\varepsilon=0\;\mathrm{(baseline)}", dir_bl, "modes_eps0.000_alpha0.00_tau0.00.h5", :gray50),
        (L"\varepsilon=0.01", dir_a, "modes_eps0.010_alpha0.00_tau0.00.h5", PALETTE[1]),
        (L"\varepsilon=0.05", dir_a, "modes_eps0.050_alpha0.00_tau0.00.h5", PALETTE[2]),
        (L"\varepsilon=0.10", dir_a, "modes_eps0.100_alpha0.00_tau0.00.h5", PALETTE[3]),
        (L"\varepsilon=0.20", dir_a, "modes_eps0.200_alpha0.00_tau0.00.h5", PALETTE[4]),
    ])
        f = joinpath(dir, file)
        isfile(f) || continue
        d = load_point(f)
        ti = argmin(abs.(d.t .- t_eval))
        Rn = d.rho[1:7, ti] ./ max(d.rho[1, ti], 1e-15)
        scatter!(ax, ns, Rn, color=col, marker=MARKERS[min(i, 6)], markersize=9,
                 label=label)
        lines!(ax, ns, Rn, color=col, linewidth=1.2)
    end

    axislegend(ax, position=:rt, framevisible=false, labelsize=10)
    save(joinpath(FIG_DIR, "fig_p7_spectrum.pdf"), fig)
    println("Saved fig_p7_spectrum.pdf")
end

# ─── Fig 2: |ρ̂₁|(t) growth curves ──────────────────────────────────────────

function fig_mode_growth()
    dir_a = "results/p7_scan_a_20260520_164805"
    dir_bl = "results/p7_scan_baseline_20260520_164748"

    fig = Figure(size=(560, 360), fontsize=12)
    ax = Axis(fig[1, 1],
        xlabel=L"t",
        ylabel=L"|\hat\rho_1(t)|",
        limits=(0, 82, -0.02, 1.05))

    for (i, (label, dir, file, col)) in enumerate([
        (L"\varepsilon=0", dir_bl, "modes_eps0.000_alpha0.00_tau0.00.h5", :gray50),
        (L"\varepsilon=0.01", dir_a, "modes_eps0.010_alpha0.00_tau0.00.h5", PALETTE[1]),
        (L"\varepsilon=0.05", dir_a, "modes_eps0.050_alpha0.00_tau0.00.h5", PALETTE[2]),
        (L"\varepsilon=0.10", dir_a, "modes_eps0.100_alpha0.00_tau0.00.h5", PALETTE[3]),
        (L"\varepsilon=0.20", dir_a, "modes_eps0.200_alpha0.00_tau0.00.h5", PALETTE[4]),
    ])
        f = joinpath(dir, file)
        isfile(f) || continue
        d = load_point(f)
        lines!(ax, d.t, d.rho[1, :], color=col, linewidth=1.8, label=label)
    end

    axislegend(ax, position=:rb, framevisible=false, labelsize=10)
    save(joinpath(FIG_DIR, "fig_p7_rho1_growth.pdf"), fig)
    println("Saved fig_p7_rho1_growth.pdf")
end

# ─── Fig 3: R₂(ε) transition at τ_r=0 ─────────────────────────────────────

function fig_R2_vs_eps()
    dir_a = "results/p7_scan_a_20260520_164805"
    dir_bl = "results/p7_scan_baseline_20260520_164748"

    fig = Figure(size=(440, 320), fontsize=12)
    ax = Axis(fig[1, 1],
        xlabel=L"\varepsilon",
        ylabel=L"R_2(\tau_r = 0)",
        limits=(-0.01, 0.22, 0, 1.05))

    eps_vals = Float64[]
    R2_vals = Float64[]

    bl = joinpath(dir_bl, "modes_eps0.000_alpha0.00_tau0.00.h5")
    if isfile(bl)
        d = load_point(bl)
        push!(eps_vals, 0.0)
        rho = d.rho
        tp = argmin(abs.(d.t .- 5.0))
        win = findall(2.0 .<= d.t .<= 8.0)
        tp2 = isempty(win) ? tp : win[argmax(rho[1, win])]
        push!(R2_vals, rho[2, tp2] / max(rho[1, tp2], 1e-15))
    end

    for ε_str in ["0.010", "0.050", "0.100", "0.200"]
        f = joinpath(dir_a, "modes_eps$(ε_str)_alpha0.00_tau0.00.h5")
        isfile(f) || continue
        d = load_point(f)
        rho = d.rho
        win = findall(2.0 .<= d.t .<= 8.0)
        tp = isempty(win) ? argmax(rho[1,:]) : win[argmax(rho[1, win])]
        push!(eps_vals, d.ε)
        push!(R2_vals, rho[2, tp] / max(rho[1, tp], 1e-15))
    end

    scatter!(ax, eps_vals, R2_vals, color=:firebrick, markersize=12)
    lines!(ax, eps_vals, R2_vals, color=:firebrick, linewidth=2)
    hlines!(ax, [0.16], color=:gray50, linestyle=:dot, linewidth=1,
            label=L"\mathrm{elastic\;baseline}\;R_2 \approx 0.16")
    hlines!(ax, [0.5], color=:gray70, linestyle=:dash, linewidth=0.8,
            label=L"R_2 = 0.5")

    axislegend(ax, position=:rb, framevisible=false, labelsize=10)
    save(joinpath(FIG_DIR, "fig_p7_R2_vs_eps.pdf"), fig)
    println("Saved fig_p7_R2_vs_eps.pdf")
end

# ─── Fig 4: Combined (ε, α) heatmap at τ_r=0 ──────────────────────────────

function fig_combined_heatmap()
    dir = "results/p7_scan_ab_20260520_171843"
    isdir(dir) || return

    data = Dict{Tuple{Float64,Float64}, Float64}()
    for f in readdir(dir; join=true)
        endswith(f, ".h5") || continue
        d = load_point(f)
        d.τ ≈ 0.0 || continue
        rho = d.rho
        win = findall(2.0 .<= d.t .<= 8.0)
        tp = isempty(win) ? argmax(rho[1,:]) : win[argmax(rho[1, win])]
        data[(d.ε, d.α)] = rho[2, tp] / max(rho[1, tp], 1e-15)
    end

    isempty(data) && return

    eps_vals = sort(unique(first.(keys(data))))
    alp_vals = sort(unique(last.(keys(data))))

    R2_mat = [get(data, (e, a), NaN) for e in eps_vals, a in alp_vals]

    fig = Figure(size=(480, 340), fontsize=12)
    ax = Axis(fig[1, 1],
        xlabel=L"\varepsilon",
        ylabel=L"\alpha",
        xticks=(1:length(eps_vals), [latexstring(@sprintf("%.2f", e)) for e in eps_vals]),
        yticks=(1:length(alp_vals), [latexstring(@sprintf("%.2f", a)) for a in alp_vals]))

    hm = heatmap!(ax, 1:length(eps_vals), 1:length(alp_vals), R2_mat,
                  colormap=:inferno, colorrange=(0, 1.0))

    for (ie, e) in enumerate(eps_vals)
        for (ia, a) in enumerate(alp_vals)
            v = R2_mat[ie, ia]
            isnan(v) && continue
            text!(ax, ie, ia, text=@sprintf("%.2f", v),
                  align=(:center, :center), fontsize=9,
                  color=v > 0.5 ? :white : :black)
        end
    end

    Colorbar(fig[1, 2], hm, label=L"R_2(\tau_r=0)")
    save(joinpath(FIG_DIR, "fig_p7_heatmap_eps_alpha.pdf"), fig)
    println("Saved fig_p7_heatmap_eps_alpha.pdf")
end

# ─── Run all ────────────────────────────────────────────────────────────────

fig_spectrum()
fig_mode_growth()
fig_R2_vs_eps()
fig_combined_heatmap()

println("\nDone. All physics figures in $FIG_DIR")
