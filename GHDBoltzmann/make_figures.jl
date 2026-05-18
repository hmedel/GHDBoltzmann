#=
Generate the four essential figures for Paper V.

Fig 1: λ_T(m_max) on log-log — the integrability-to-diffusion crossover
Fig 2: |ρ̂_n(t)| step-IC trajectories
Fig 3: Mode amplitude ratio |ρ̂_n|/|ρ̂_1| vs n, linear and nonlinear
Fig 4: ω(k) wavenumber sweep showing hydro→kinetic crossover

Outputs PDFs in Paper02/figures/
=#
push!(LOAD_PATH, @__DIR__)
using Plots, LaTeXStrings
using DelimitedFiles
using Printf

gr()
default(framestyle=:box, gridstyle=:dot, gridlinewidth=0.4,
        size=(450, 330), titlefontsize=11, labelfontsize=10,
        legendfontsize=8, tickfontsize=8)

const FIGDIR = joinpath(@__DIR__, "..", "Paper02", "figures")
isdir(FIGDIR) || mkpath(FIGDIR)

# ----------------------------------------------------------------------
# Figure 1: λ_T(m_max) integrability-to-diffusion crossover
# ----------------------------------------------------------------------
println("[Fig 1] λ_T(m_max) log-log + power-law fit")
# Data from convergence_and_scaling.jl
m_vals = [1.01, 1.1, 1.5, 2.0, 5.0, 10.0, 20.0, 100.0]
λ_vals = [9.68e6, 1.02e5, 5356.0, 1796.0, 378.0, 235.0, 186.0, 79.0]

# Power-law fit on m_max ∈ [1.1, 5]: log λ = a + α log(m-1) (α<0)
fit_idx = findall(m -> 1.05 < m < 6, m_vals)
x_fit = log10.(m_vals[fit_idx] .- 1)
y_fit = log10.(λ_vals[fit_idx])
α_slope = sum((x_fit .- sum(x_fit)/length(x_fit)) .*
              (y_fit .- sum(y_fit)/length(y_fit))) /
          sum((x_fit .- sum(x_fit)/length(x_fit)).^2)
y_intercept = sum(y_fit)/length(y_fit) - α_slope * sum(x_fit)/length(x_fit)
@printf("  fitted slope α = %.3f, intercept = %.3f\n", α_slope, y_intercept)

mm = 10 .^ (range(-2.5, 2.5, length=100))
λ_fit = (10^y_intercept) .* mm .^ α_slope

p1 = plot(xscale=:log10, yscale=:log10,
          xlabel=L"m_{\max} - 1", ylabel=L"\lambda_T",
          xlim=(0.005, 200), ylim=(20, 5e7),
          legend=:topright)
plot!(mm, λ_fit, color=:gray, ls=:dash, lw=1.5,
      label=L"\sim (m_{\max}-1)^{-1.6}")
scatter!(m_vals[2:end] .- 1, λ_vals[2:end], ms=6,
         color=:red, label="Chapman--Enskog")
# Annotate plateau
hline!([79], color=:blue, ls=:dot, lw=1, label=L"\lambda_T \approx 80\ (m_{\max}\to\infty)")
savefig(p1, joinpath(FIGDIR, "fig_lambdaT_crossover.pdf"))
println("  saved fig_lambdaT_crossover.pdf")

# ----------------------------------------------------------------------
# Figure 2: |ρ̂_n(t)| step-IC trajectories from saved CSV
# ----------------------------------------------------------------------
println("[Fig 2] mode trajectories from step IC")
csv_lin = joinpath(@__DIR__, "step_sim_step_U005.csv")
if isfile(csv_lin)
    data = readdlm(csv_lin, ',', skipstart=1)
    t = data[:, 1]
    T_sound = 11.31
    p2 = plot(xlabel=L"t/T_{\rm sound}", ylabel=L"|\hat\rho_n(t)|",
              yscale=:log10, ylim=(1e-6, 0.1),
              legend=:bottomleft, legendcolumns=2)
    colors = [:black, :red, :blue, :green, :purple, :orange, :brown]
    for n in 1:7
        plot!(t/T_sound, max.(data[:, n+1], 1e-7),
              label="n=$n", color=colors[n], lw=1.2)
    end
    title!(L"U_0 = 0.05,\ m_{\max} = 5")
    savefig(p2, joinpath(FIGDIR, "fig_mode_trajectories.pdf"))
    println("  saved fig_mode_trajectories.pdf")
else
    println("  skipped (no CSV)")
end

# ----------------------------------------------------------------------
# Figure 3: |ρ̂_n|/|ρ̂_1| vs n for linear and nonlinear
# ----------------------------------------------------------------------
println("[Fig 3] mode-amplitude ratios linear vs nonlinear")
# First-peak data (from analyze_step.jl output)
n_arr = [1, 2, 3, 4, 5, 6, 7]
ratio_lin  = [1.0000, 0.0049, 0.3147, 0.0025, 0.1803, 0.0016, 0.1170]
ratio_nl   = [1.0000, 0.0432, 0.3167, 0.0214, 0.1774, 0.0134, 0.1213]
inv_n      = 1.0 ./ n_arr

p3 = plot(xlabel=L"n", ylabel=L"|\hat\rho_n|/|\hat\rho_1|",
          xlim=(0.5, 7.5),
          legend=:topright,
          yscale=:log10, ylim=(1e-3, 1.5))
plot!(n_arr, inv_n, color=:gray, lw=1.5, ls=:dash,
      label=L"1/n\ (\rm linear\ prediction)")
scatter!(n_arr, ratio_lin, ms=7, color=:red, marker=:circle,
         label=L"U_0 = 0.05")
scatter!(n_arr, ratio_nl, ms=7, color=:blue, marker=:square,
         label=L"U_0 = 0.5\ (\rm nonlinear)")
title!(L"m_{\max} = 5,\ {\rm first\text{-}peak\ amplitude}")
savefig(p3, joinpath(FIGDIR, "fig_mode_ratios.pdf"))
println("  saved fig_mode_ratios.pdf")

# ----------------------------------------------------------------------
# Figure 4: ω(k) hydro→kinetic crossover (from Table omega-k in Sec VII)
# ----------------------------------------------------------------------
println("[Fig 4] wavenumber sweep showing hydro→kinetic crossover")
k_arr = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
γ_eff_arr = [10.5, 7.3, 2.21, 0.346, 0.086, 0.021]
Im_ω_arr  = [0.00105, 0.0183, 0.0221, 0.0216, 0.0214, 0.0214]

p4a = plot(xscale=:log10, yscale=:log10,
           xlabel=L"k", ylabel=L"\gamma_{\rm eff} = -{\rm Im}\,\omega/k^2",
           legend=:topright, xlim=(0.005, 2), ylim=(0.01, 30))
scatter!(k_arr, γ_eff_arr, ms=6, color=:red,
         label="reduced 4D projection")
hline!([0.021], color=:gray, ls=:dot, label=L"\tau_{\rm break}^{-1}/k_1^2")
title!(L"\gamma_{\rm eff}(k):\ \mathrm{Kn}\sim 1\ \mathrm{crossover}")
savefig(p4a, joinpath(FIGDIR, "fig_omega_k_gamma.pdf"))
println("  saved fig_omega_k_gamma.pdf")

p4b = plot(xscale=:log10, yscale=:log10,
           xlabel=L"k", ylabel=L"|{\rm Im}\,\omega|",
           legend=:topleft, xlim=(0.005, 2), ylim=(5e-4, 0.1))
scatter!(k_arr, Im_ω_arr, ms=6, color=:blue, label=L"|{\rm Im}\,\omega_{\rm sound}|")
plot!(k_arr, 10 .* k_arr.^2, color=:gray, ls=:dash,
      label=L"\gamma_{\rm CE}k^2\ (\mathrm{hydro})")
hline!([0.05], color=:gray, ls=:dot,
       label=L"\tau_{\rm break}^{-1}\ (\mathrm{kinetic})")
title!(L"|{\rm Im}\,\omega(k)|:\ \mathrm{saturation\ at\ collision\ rate}")
savefig(p4b, joinpath(FIGDIR, "fig_omega_k_Im.pdf"))
println("  saved fig_omega_k_Im.pdf")

println("\nAll figures generated in $FIGDIR")
