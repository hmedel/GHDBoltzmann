#=
Extended L sweep for hyperscaling test.

Refinements vs heat_autocorrelator.jl:
  - Skip L=6 (sound/heat hybridisation, no clean decay)
  - Extend to L=150 (~ 20 ℓ_mfp, deep hydro)
  - N_s = round(Int, 4*L), capped at 400, to keep runtime feasible
  - T_obs = max(30, 4·τ_κ) where τ_κ = 1/(γ_heat est) — adaptive
  - Multi-period fit: skip first τ_sound/2 to avoid IC transient
  - Save raw |Ĵ_E(t)| traces for each (L, m) to allow off-line fits
=#
push!(LOAD_PATH, @__DIR__)
using GHDBoltzmannSolver
using Printf, Statistics, DelimitedFiles

const sigma_val = 0.00314
const rho_tot_target = 31.822
const ε_excite = 1e-3
const T0 = 1.0

function init_heat_excitation!(st, m_max, ε, k, ρ_tot)
    p = st.p
    Z_L = sqrt(2π); Z_H = sqrt(2π/m_max)
    @inbounds for ki in 1:p.N_s, j in 1:p.N_v
        s = st.s[ki]; v = st.v[j]
        cos_ks = cos(k*s)
        f_L = (ρ_tot/2 / Z_L) * exp(-v^2/2)
        f_H = (ρ_tot/2 / Z_H) * exp(-m_max*v^2/2)
        J_E_L = v * (v^2/2 - 1.5*T0)
        J_E_H = v * (m_max*v^2/2 - 1.5*T0)
        st.rho_L[ki, j] = f_L * (1.0 + ε * J_E_L * cos_ks)
        st.rho_H[ki, j] = f_H * (1.0 + ε * J_E_H * cos_ks)
    end
end

function heat_current_fourier(st, m_max, k)
    p = st.p
    a = 0.0
    @inbounds for ki in 1:p.N_s
        J_density = 0.0
        for j in 1:p.N_v
            v = st.v[j]
            J_density += v * (v^2/2 - 1.5*T0) * st.rho_L[ki, j] * st.dv
            J_density += v * (m_max*v^2/2 - 1.5*T0) * st.rho_H[ki, j] * st.dv
        end
        a += cos(k*st.s[ki]) * J_density
    end
    return 2*a/p.N_s
end

function fit_log_decay(times, Js; t_skip=2.0)
    # Fit log|J| = a + b·t on |J| values > threshold, t > t_skip
    valid = findall(i -> abs(Js[i]) > 1e-12 && times[i] > t_skip, 1:length(Js))
    n = length(valid)
    if n < 4
        return NaN, NaN
    end
    t_fit = times[valid]; lJ = log.(abs.(Js[valid]))
    mt = sum(t_fit)/n; mL = sum(lJ)/n
    slope = sum((t_fit .- mt).*(lJ .- mL))/sum((t_fit .- mt).^2)
    return -slope, n  # γ_heat = -slope
end

function autocorrelator_run(L::Float64, m_max::Float64; T_obs=30.0)
    N_rods = round(Int, rho_tot_target * L)
    ρ_tot = N_rods / L
    η = sigma_val * ρ_tot
    k1 = 2π/L
    N_s = min(400, max(40, round(Int, 4*L)))
    p = GHDBParams(L=L, sigma=sigma_val, m_L=1.0, m_H=m_max, kT=1.0,
                   N_s=N_s, N_v=96, V_max=6.0, cfl=0.25)
    st = GHDBState(p)
    init_heat_excitation!(st, m_max, ε_excite, k1, ρ_tot)
    J0 = heat_current_fourier(st, m_max, k1)
    times = Float64[]; Js = Float64[]
    push!(times, 0.0); push!(Js, J0)
    cb(s) = heat_current_fourier(s, m_max, k1)
    records, _, _ = run_simulation!(st, T_obs, save_every=0.25, callback=cb)
    for r in records[2:end]
        push!(times, r[1]); push!(Js, r[2])
    end
    # Skip first τ_sound/2 to avoid IC transient
    τ_sound = L * 0.5  # rough estimate of half sound period (c_s ~ 1)
    γ, npts = fit_log_decay(times, Js; t_skip=τ_sound/4)
    κ = γ / k1^2
    return γ, κ, J0, ρ_tot, η, N_s, times, Js
end

println("="^70)
println("Extended L×m_max sweep")
println("="^70)

L_grid = [12.57, 18.0, 25.0, 35.0, 50.0, 75.0, 100.0]
m_grid = [1.5, 2.0, 3.0, 5.0, 10.0]
results = Dict{Tuple{Float64,Float64}, NamedTuple}()
total_runs = length(L_grid)*length(m_grid)
i_run = 0
t_start = time()
for (i, L) in enumerate(L_grid), (j, m) in enumerate(m_grid)
    global i_run += 1
    t_run = @elapsed begin
        γ, κ, J0, ρ_tot, η, N_s, times, Js = autocorrelator_run(L, m)
        results[(L, m)] = (γ=γ, κ=κ, J0=J0, ρ_tot=ρ_tot, η=η, N_s=N_s)
    end
    elapsed = time() - t_start
    @printf("[%2d/%2d] L=%6.2f m=%4.1f  γ=%+9.4e  κ=%+9.4e  N_s=%3d  [run %.1fs, total %.1fs]\n",
            i_run, total_runs, L, m, γ, κ, results[(L, m)].N_s, t_run, elapsed)
end

println("\n"*"="^70)
println("Summary: κ_eff(L, m_max) = γ_heat/k_1²")
println("="^70)
print("L\\m_max\t")
for m in m_grid; print(@sprintf("%6.1f\t", m)); end
println()
for L in L_grid
    print(@sprintf("L=%5.2f\t", L))
    for m in m_grid
        κ = results[(L, m)].κ
        print(@sprintf("%6.3f\t", κ))
    end
    println()
end

# Hyperscaling test
const β = 1/3
const ζ = 0.218
println("\nHyperscaling test: λ_T·k^β vs (m-1)k^{-ζ}  [β=1/3, ζ=0.218]")
xs = Float64[]; ys = Float64[]
for (L, m) in keys(results)
    k = 2π/L; κ = results[(L, m)].κ
    if κ > 0
        push!(xs, (m-1.0)*k^(-ζ))
        push!(ys, κ * k^β)
    end
end
log_x = log10.(xs); log_y = log10.(ys)
println("\nData points (n=$(length(xs))):")
println("  log10(x) range: [$(round(minimum(log_x),digits=2)), $(round(maximum(log_x),digits=2))]")
println("  log10(y) range: [$(round(minimum(log_y),digits=2)), $(round(maximum(log_y),digits=2))]")

n_bin = 5
bins = range(minimum(log_x), maximum(log_x), length=n_bin+1)
stds = Float64[]
for b in 1:n_bin
    mask = (log_x .≥ bins[b]) .& (log_x .≤ bins[b+1] + 1e-10)
    if count(mask) >= 2
        σ = sqrt(sum((log_y[mask] .- sum(log_y[mask])/count(mask)).^2)/(count(mask)-1))
        push!(stds, σ)
        @printf("  bin [%.2f, %.2f]: n=%d, std(log10 y) = %.3f\n",
                bins[b], bins[b+1], count(mask), σ)
    end
end
if !isempty(stds)
    @printf("\nMean within-bin std: %.3f dex  (collapse threshold: ≤ 0.1)\n",
            sum(stds)/length(stds))
end

# Save raw data
csv_path = joinpath(@__DIR__, "extended_sweep.csv")
open(csv_path, "w") do io
    write(io, "L,m_max,k1,gamma_heat,kappa_eff,J0,rho_tot,eta,N_s\n")
    for L in L_grid, m in m_grid
        r = results[(L, m)]
        @printf(io, "%g,%g,%g,%g,%g,%g,%g,%g,%d\n",
                L, m, 2π/L, r.γ, r.κ, r.J0, r.ρ_tot, r.η, r.N_s)
    end
end
println("\nRaw data: extended_sweep.csv")

# Generate figure
using Plots, LaTeXStrings
gr()
default(framestyle=:box, gridstyle=:dot, size=(450, 330),
        titlefontsize=11, labelfontsize=10, legendfontsize=8, tickfontsize=8)
colors = palette(:viridis, length(m_grid))
p = plot(xscale=:log10, yscale=:log10,
         xlabel=L"(m_{\max}-1)\,k^{-\zeta}",
         ylabel=L"\kappa_{\rm eff}\,k^{\beta}",
         title=L"\rm Hyperscaling\ test\ (\beta=1/3, \zeta=0.218)",
         legend=:topleft, legendcolumns=2)
for (j, m) in enumerate(m_grid)
    xs_m = Float64[]; ys_m = Float64[]
    for L in L_grid
        k = 2π/L; κ = results[(L, m)].κ
        if κ > 0
            push!(xs_m, (m-1.0)*k^(-ζ))
            push!(ys_m, κ*k^β)
        end
    end
    if length(xs_m) > 0
        scatter!(xs_m, ys_m, ms=5, color=colors[j], label=@sprintf("m=%.1f", m))
    end
end
savefig(p, joinpath(@__DIR__, "..", "Paper02", "figures", "fig_extended_hyperscaling.pdf"))
println("Figure: fig_extended_hyperscaling.pdf")
