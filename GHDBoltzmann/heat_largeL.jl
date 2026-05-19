#=
Large-L heat-current autocorrelator for hyperscaling verification.

At L=200, k₁=2π/200≈0.031, Kn=k₁·ℓ_mfp≈0.22 — inside the
hydrodynamic regime where the heat mode should decouple from
sound and streaming modes, allowing clean γ_heat extraction.

L grid: [200, 300, 400]
m_max grid: [1.5, 2.0, 3.0, 5.0, 10.0]

Expected runtime per point: O(minutes) at N_s=4L, N_v=96.
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
    valid = findall(i -> abs(Js[i]) > 1e-12 && times[i] > t_skip, 1:length(Js))
    n = length(valid)
    if n < 6
        return NaN, NaN, Float64[], Float64[]
    end
    t_fit = times[valid]; lJ = log.(abs.(Js[valid]))
    mt = sum(t_fit)/n; mL = sum(lJ)/n
    slope = sum((t_fit .- mt).*(lJ .- mL))/sum((t_fit .- mt).^2)
    residuals = lJ .- (mL .+ slope .* (t_fit .- mt))
    return -slope, n, t_fit, residuals
end

function largeL_run(L::Float64, m_max::Float64)
    N_rods = round(Int, rho_tot_target * L)
    ρ_tot = N_rods / L
    η = sigma_val * ρ_tot
    k1 = 2π/L
    Kn = k1 * 7.0  # ℓ_mfp ≈ 7

    N_s = round(Int, 4*L)
    # Adaptive T_obs: need several heat-mode e-folding times
    # γ_heat ~ κ·k² with κ ~ λ_T/(ρ·c_p) ~ 100 for m=5
    # → γ ~ 100·k² → τ_heat ~ 1/(100·k²)
    # Be conservative: T_obs = max(80, 6/γ_est)
    κ_est = 100.0  # rough estimate
    γ_est = κ_est * k1^2
    T_obs = max(80.0, 6.0/γ_est)
    T_obs = min(T_obs, 500.0)  # cap to prevent runaway

    @printf("  Setting up: L=%.0f, N_s=%d, Kn=%.3f, T_obs=%.0f, γ_est=%.4f\n",
            L, N_s, Kn, T_obs, γ_est)

    p = GHDBParams(L=L, sigma=sigma_val, m_L=1.0, m_H=m_max, kT=1.0,
                   N_s=N_s, N_v=96, V_max=6.0, cfl=0.25)
    st = GHDBState(p)
    init_heat_excitation!(st, m_max, ε_excite, k1, ρ_tot)
    J0 = heat_current_fourier(st, m_max, k1)

    times = Float64[]; Js = Float64[]
    push!(times, 0.0); push!(Js, J0)

    save_dt = max(0.5, T_obs/400)
    cb(s) = heat_current_fourier(s, m_max, k1)
    records, _, _ = run_simulation!(st, T_obs, save_every=save_dt, callback=cb)
    for r in records[2:end]
        push!(times, r[1]); push!(Js, r[2])
    end

    # Skip initial transient: τ_sound/2 where τ_sound = L/c_s ≈ L
    t_skip = L * 0.3
    γ, npts, t_fit, residuals = fit_log_decay(times, Js; t_skip=t_skip)
    κ = γ / k1^2

    # Fit quality: RMS of residuals in log space
    rms_resid = isempty(residuals) ? NaN : sqrt(sum(residuals.^2)/length(residuals))

    return (γ=γ, κ=κ, J0=J0, ρ_tot=ρ_tot, η=η, N_s=N_s, Kn=Kn,
            T_obs=T_obs, npts=npts, rms_resid=rms_resid,
            times=times, Js=Js)
end

println("="^70)
println("Large-L heat-current autocorrelator for hyperscaling test")
println("="^70)

L_grid = [200.0, 300.0, 400.0]
m_grid = [1.5, 2.0, 3.0, 5.0, 10.0]

results = Dict{Tuple{Float64,Float64}, NamedTuple}()
total_runs = length(L_grid)*length(m_grid)
i_run = 0
t_start = time()

for L in L_grid, m in m_grid
    global i_run += 1
    @printf("\n[%2d/%2d] L=%.0f  m=%.1f\n", i_run, total_runs, L, m)
    t_run = @elapsed begin
        r = largeL_run(L, m)
        results[(L, m)] = r
    end
    elapsed = time() - t_start
    @printf("  → γ=%+10.5e  κ=%+10.5e  npts=%d  rms=%.3f  [%.1fs, total %.1fs]\n",
            r.γ, r.κ, r.npts, r.rms_resid, t_run, elapsed)

    # Save raw trace
    trace_path = joinpath(@__DIR__, "traces_largeL",
                          @sprintf("trace_L%g_m%g.csv", L, m))
    mkpath(dirname(trace_path))
    open(trace_path, "w") do io
        write(io, "t,J_E\n")
        for i in 1:length(r.times)
            @printf(io, "%.6f,%.10e\n", r.times[i], r.Js[i])
        end
    end
end

# Summary table
println("\n" * "="^70)
println("Summary: κ_eff = γ_heat / k₁²")
println("="^70)
@printf("%-8s", "L\\m")
for m in m_grid; @printf("%10.1f", m); end
println()
for L in L_grid
    @printf("L=%-5.0f", L)
    for m in m_grid
        r = results[(L, m)]
        if isnan(r.κ)
            @printf("%10s", "---")
        else
            @printf("%10.2f", r.κ)
        end
    end
    println()
end

# Fit quality
println("\nFit quality (RMS of log-residuals, < 0.1 = clean exponential):")
@printf("%-8s", "L\\m")
for m in m_grid; @printf("%10.1f", m); end
println()
for L in L_grid
    @printf("L=%-5.0f", L)
    for m in m_grid
        r = results[(L, m)]
        @printf("%10.3f", r.rms_resid)
    end
    println()
end

# Hyperscaling test
const β = 1/3
const α_exp = 1.53
const ζ = β / α_exp

println("\n" * "="^70)
println("Hyperscaling test: κ·k^β vs (m-1)·k^{-ζ}")
@printf("β=1/3, α=%.2f, ζ=%.3f\n", α_exp, ζ)
println("="^70)

xs = Float64[]; ys = Float64[]; tags = String[]
for L in L_grid, m in m_grid
    r = results[(L, m)]
    k = 2π/L
    if !isnan(r.κ) && r.κ > 0 && r.rms_resid < 0.3
        push!(xs, (m-1.0)*k^(-ζ))
        push!(ys, r.κ * k^β)
        push!(tags, @sprintf("L=%.0f,m=%.1f", L, m))
    end
end

if length(xs) >= 4
    log_x = log10.(xs); log_y = log10.(ys)
    @printf("\nData points with clean fits: %d / %d\n", length(xs), total_runs)
    @printf("log₁₀(x) range: [%.2f, %.2f]\n", minimum(log_x), maximum(log_x))
    @printf("log₁₀(y) range: [%.2f, %.2f]\n", minimum(log_y), maximum(log_y))

    n_bins = 4
    bins = range(minimum(log_x), maximum(log_x), length=n_bins+1)
    stds = Float64[]
    for b in 1:n_bins
        mask = (log_x .≥ bins[b]) .& (log_x .≤ bins[b+1] + 1e-10)
        nm = count(mask)
        if nm >= 2
            μ = sum(log_y[mask])/nm
            σ = sqrt(sum((log_y[mask] .- μ).^2)/(nm-1))
            push!(stds, σ)
            @printf("  bin [%.2f, %.2f]: n=%d, std=%.3f dex\n",
                    bins[b], bins[b+1], nm, σ)
        end
    end
    if !isempty(stds)
        mean_std = sum(stds)/length(stds)
        @printf("\nMean within-bin std: %.3f dex\n", mean_std)
        if mean_std < 0.1
            println(">>> HYPERSCALING VERIFIED (collapse < 0.1 dex) <<<")
        elseif mean_std < 0.2
            println(">>> Marginal collapse (0.1-0.2 dex) — suggestive but not definitive <<<")
        else
            println(">>> No collapse (> 0.2 dex) — hyperscaling not verified <<<")
        end
    end
else
    println("\nInsufficient clean data points for hyperscaling test.")
end

# Save summary CSV
csv_path = joinpath(@__DIR__, "largeL_sweep.csv")
open(csv_path, "w") do io
    write(io, "L,m_max,k1,Kn,gamma_heat,kappa_eff,rms_resid,npts,T_obs\n")
    for L in L_grid, m in m_grid
        r = results[(L, m)]
        @printf(io, "%g,%g,%.6f,%.3f,%.6e,%.6e,%.4f,%d,%.1f\n",
                L, m, 2π/L, r.Kn, r.γ, r.κ, r.rms_resid, r.npts, r.T_obs)
    end
end
println("\nRaw data: largeL_sweep.csv")
println("Traces:   traces_largeL/")
