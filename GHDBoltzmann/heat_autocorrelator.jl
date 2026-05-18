#=
Time-resolved heat-current relaxation via DVM.

IC: ρ_α(s,v,0) = ρ_α^(0)(v) [1 + ε J_E^(α)(v) cos(ks)]
with J_E^(α)(v) = v(m_α v²/2 - 3T/2).

By parity (J_E odd in v):
  δn_α = ∫ δρ dv = 0    (mass unperturbed)
  δp = ∫ v m δρ dv = ∫ v² (mv²/2 - 3T/2) m ρ^(0) cos ks dv = 0
  δe = (1/2) ∫ m v² δρ dv = 0  (odd integral)

Only the heat-current at wavenumber k is excited.

Evolution: track Ĵ_E(k, t) = (1/L) Σ_s Δs cos(ks) · ⟨v(m v²/2 - 3T/2)⟩(s,t)
Fit |Ĵ_E(t)| ~ e^{-γ_heat t}. Extract κ_eff = γ_heat/k².

Sweep: (L, m_max) grid → test hyperscaling.
=#
push!(LOAD_PATH, @__DIR__)
using GHDBoltzmannSolver
using Printf, Statistics

const sigma_val = 0.00314
const rho_tot_target = 31.822  # match P3 reference (independent of L)
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
    # Compute Ĵ_E(k, t) = (2/N_s) Σ_s cos(ks) · J_E_density(s)
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

function autocorrelator_run(L::Float64, m_max::Float64; T_obs=20.0)
    # Match P3 density independent of L → vary N (number of rods)
    N_rods = round(Int, rho_tot_target * L)
    ρ_tot = N_rods / L
    η = sigma_val * ρ_tot
    k1 = 2π/L
    # Time scale
    τ_break_est = 1/(ρ_tot * sigma_val * 1.1)
    # Grid: N_s scales with L
    N_s = max(40, round(Int, 6*L))
    p = GHDBParams(L=L, sigma=sigma_val, m_L=1.0, m_H=m_max, kT=1.0,
                   N_s=N_s, N_v=96, V_max=6.0, cfl=0.25)
    st = GHDBState(p)
    init_heat_excitation!(st, m_max, ε_excite, k1, ρ_tot)
    J0 = heat_current_fourier(st, m_max, k1)

    times = Float64[]; Js = Float64[]
    push!(times, 0.0); push!(Js, J0)

    cb(s) = heat_current_fourier(s, m_max, k1)
    records, _, _ = run_simulation!(st, T_obs, save_every=0.2, callback=cb)
    for r in records[2:end]
        push!(times, r[1]); push!(Js, r[2])
    end

    # Fit exponential decay |J(t)| = J0 exp(-γ t)
    valid_idx = findall(j -> abs(j) > 1e-15, Js)
    log_J = log.(abs.(Js[valid_idx]))
    t_fit = times[valid_idx]
    # Linear regression of log|J| vs t, weighted toward early times
    n_pts = length(t_fit)
    if n_pts < 4
        return NaN, NaN, J0, ρ_tot, η
    end
    mt = sum(t_fit)/n_pts; mL = sum(log_J)/n_pts
    slope = sum((t_fit .- mt).*(log_J .- mL)) / sum((t_fit .- mt).^2)
    γ_heat = -slope
    κ_eff = γ_heat / k1^2
    return γ_heat, κ_eff, J0, ρ_tot, η
end

println("="^70)
println("Heat-current autocorrelator via DVM (linear-response)")
println("="^70)

L_grid = [6.0, 12.57, 25.0, 50.0]
m_grid = [1.5, 2.0, 5.0, 10.0]

println("\nL\\m_max\t" * join([@sprintf("%.1f", m) for m in m_grid], "\t"))
println("γ_heat values:")
γ_data = zeros(length(L_grid), length(m_grid))
κ_data = zeros(length(L_grid), length(m_grid))
for (i, L) in enumerate(L_grid)
    print(@sprintf("L=%.2f\t", L))
    for (j, m) in enumerate(m_grid)
        γ, κ, J0, ρ_tot, η = autocorrelator_run(L, m)
        γ_data[i, j] = γ
        κ_data[i, j] = κ
        print(@sprintf("%.3e\t", γ))
    end
    println()
end
println("\nκ_eff = γ_heat/k_1² values:")
println("L\\m_max\t" * join([@sprintf("%.1f", m) for m in m_grid], "\t"))
for (i, L) in enumerate(L_grid)
    print(@sprintf("L=%.2f\t", L))
    for (j, m) in enumerate(m_grid)
        print(@sprintf("%.3f\t", κ_data[i, j]))
    end
    println()
end

# Hyperscaling test
println("\nHyperscaling: collapse of κ·k^β vs (m-1)·k^{-ζ}")
println("β=1/3, ζ=0.218 (predicted)")
println()
xs = Float64[]; ys = Float64[]
for (i, L) in enumerate(L_grid), (j, m) in enumerate(m_grid)
    k = 2π/L
    if isfinite(κ_data[i, j]) && κ_data[i, j] > 0
        push!(xs, (m-1.0)*k^(-0.218))
        push!(ys, κ_data[i, j]*k^(1/3))
        @printf("  L=%.2f m=%.1f  k=%.3f  κ=%.3f  →  x=%.3f, y=%.3f\n",
                L, m, k, κ_data[i, j], (m-1.0)*k^(-0.218),
                κ_data[i, j]*k^(1/3))
    end
end

if length(xs) >= 4
    log_x = log10.(xs); log_y = log10.(ys)
    println("\nLog-log spread: x=[$(round(minimum(log_x),digits=2)), $(round(maximum(log_x),digits=2))]")
    println("                y=[$(round(minimum(log_y),digits=2)), $(round(maximum(log_y),digits=2))]")
    # Bin by x
    n_bin = 3
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
        @printf("\nMean within-bin std: %.3f dex\n", sum(stds)/length(stds))
        println("(collapse ≤ 0.1 dex → hyperscaling verified)")
    end
end
