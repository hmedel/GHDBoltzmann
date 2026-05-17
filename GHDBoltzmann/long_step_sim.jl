#=
Long step-velocity-IC simulation for Paper V Sec VI.

IC: u(s) = U_0 sgn(cos(k_1 s)), local Maxwellian background.
Track Fourier amplitudes |ρ̂_n(t)| for n=1..7 over many sound periods.

Output: CSV with columns (t, |ρ̂_1|, ..., |ρ̂_7|) and per-n peaks.

Parameters match P3 reference (L=12.57, σ=0.00314, ρ=31.8, η=0.1).
Run at m_max=5 (linear regime: c_s_an≈1.111, T_sound≈11.3).
Total time: 12 sound periods ≈ 135.
=#
push!(LOAD_PATH, @__DIR__)
using GHDBoltzmannSolver
using Printf, Statistics, DelimitedFiles

const L_arc   = 12.57
const sigma   = 0.00314
const rho_tot = 400/L_arc
const m_max   = 5.0
const m_bar   = (1 + m_max)/2
const eta     = sigma*rho_tot
const c_s_an  = sqrt(3/m_bar)/(1 - eta)
const k1      = 2π/L_arc
const T_sound = 2π/(c_s_an*k1)
const N_s     = 80
const N_v     = 128
const V_max   = 6.0
const N_periods = 12
const N_modes = 7

# Amplitude sweep: linear (A=0.05) and step (U_0=0.5)
function run_sim(label, init_fn, T_total)
    println("="^70)
    println("Run: $label, T_total=$(round(T_total, digits=2)) (≈ $(round(T_total/T_sound, digits=1)) periods)")
    println("c_s_an = $(round(c_s_an, digits=4)), T_sound = $(round(T_sound, digits=2))")
    println("="^70)
    p = GHDBParams(L=L_arc, sigma=sigma, m_L=1.0, m_H=m_max, kT=1.0,
                   N_s=N_s, N_v=N_v, V_max=V_max, cfl=0.25)
    st = GHDBState(p)
    init_fn(st)

    # Fourier amplitudes of density (n_L + n_H)/ρ_tot - 1
    s_grid = st.s
    cos_n = [cos.(n*k1*s_grid) for n in 1:N_modes]
    sin_n = [sin.(n*k1*s_grid) for n in 1:N_modes]

    function fourier_amps(s)
        nL = vec(sum(s.rho_L, dims=2))*s.dv
        nH = vec(sum(s.rho_H, dims=2))*s.dv
        delta = (nL .+ nH) ./ rho_tot .- 1.0
        amps = zeros(N_modes)
        for n in 1:N_modes
            a = 2.0*sum(delta .* cos_n[n])/N_s
            b = 2.0*sum(delta .* sin_n[n])/N_s
            amps[n] = sqrt(a^2 + b^2)
        end
        return amps
    end

    save_every = T_sound/40  # 40 samples per period
    t_wall = @elapsed records, n_steps, dt_actual = run_simulation!(
        st, T_total, save_every=save_every, callback=fourier_amps)
    println("Wall: $(round(t_wall, digits=1)) s, dt=$(round(dt_actual, digits=5)), n_steps=$n_steps")
    println("Samples: $(length(records))")

    # Save records
    times = [r[1] for r in records]
    amps_table = hcat(times, [hcat([r[2][n] for r in records]...)' for n in 1:N_modes]...)
    data = zeros(length(records), N_modes+1)
    for (i, r) in enumerate(records)
        data[i, 1] = r[1]
        for n in 1:N_modes
            data[i, n+1] = r[2][n]
        end
    end
    fname = "step_sim_$(label).csv"
    open(joinpath(@__DIR__, fname), "w") do io
        write(io, "t," * join(["rho_$n" for n in 1:N_modes], ",") * "\n")
        writedlm(io, data, ',')
    end
    println("Saved: $fname")

    # Peak amplitudes per mode (max over time, excluding t=0 which has IC structure)
    println("\nPer-mode peak |ρ̂_n| / |ρ̂_1| over t > T/4:")
    cutoff_idx = findfirst(t -> t > T_sound/4, times)
    cutoff_idx = something(cutoff_idx, 2)
    println("(skip first T/4 to avoid IC transient)")
    peak_1 = maximum(data[cutoff_idx:end, 2])
    for n in 1:N_modes
        peak_n = maximum(data[cutoff_idx:end, n+1])
        @printf("  n=%d:  peak=%.5e   peak/peak_1=%.4f   (1/n=%.4f)\n",
                n, peak_n, peak_n/peak_1, 1.0/n)
    end

    # Fit damping of |ρ̂_1|: extract envelope from local maxima
    rho1 = data[:, 2]
    # Find local maxima
    maxima_t = Float64[]
    maxima_v = Float64[]
    for i in 2:length(rho1)-1
        if rho1[i] > rho1[i-1] && rho1[i] > rho1[i+1] && times[i] > T_sound/4
            push!(maxima_t, times[i])
            push!(maxima_v, rho1[i])
        end
    end
    if length(maxima_t) >= 3
        log_v = log.(maxima_v)
        # Linear fit log_v = a + b*t, damping = -b
        t_centered = maxima_t .- mean(maxima_t)
        log_centered = log_v .- mean(log_v)
        γ = -sum(t_centered .* log_centered)/sum(t_centered.^2)
        @printf("\nDamping rate γ of |ρ̂_1| from %d local maxima: γ = %.5f\n",
                length(maxima_t), γ)
        @printf("  γ·k1²-implied D: D_apparent = γ/k1² = %.4f\n", γ/k1^2)
        @printf("  Quality factor Q = ω/(2γ) = %.2f\n", (c_s_an*k1)/(2*γ))
    else
        println("\nToo few maxima ($(length(maxima_t))) for damping fit.")
    end

    return data, times
end

# IC 1: linear sound eigenmode (validation already in Sec IV)
function init_eigenmode_05!(st)
    A = 0.05
    n_L_p = [(rho_tot/2)*(1 + A*cos(k1*st.s[k])) for k in 1:N_s]
    n_H_p = [(rho_tot/2)*(1 + A*cos(k1*st.s[k])) for k in 1:N_s]
    u_p   = [A*c_s_an*cos(k1*st.s[k]) for k in 1:N_s]
    init_local_maxwell!(st, n_L_p, n_H_p, u_p, 1.0)
end

# IC 2: step velocity (P3 nonlinear)
function init_step_U05!(st)
    U_0 = 0.5
    u_p = [cos(k1*st.s[k]) > 0 ? U_0 : -U_0 for k in 1:N_s]
    n_L_p = fill(rho_tot/2, N_s)
    n_H_p = fill(rho_tot/2, N_s)
    init_local_maxwell!(st, n_L_p, n_H_p, u_p, 1.0)
end

# IC 3: small-amplitude step (linear regime, tests 1/n ratio)
function init_step_U005!(st)
    U_0 = 0.05
    u_p = [cos(k1*st.s[k]) > 0 ? U_0 : -U_0 for k in 1:N_s]
    n_L_p = fill(rho_tot/2, N_s)
    n_H_p = fill(rho_tot/2, N_s)
    init_local_maxwell!(st, n_L_p, n_H_p, u_p, 1.0)
end

T_total = N_periods * T_sound

# Run all three
data1, t1 = run_sim("eigenmode_A05", init_eigenmode_05!, T_total)
data2, t2 = run_sim("step_U005", init_step_U005!, T_total)
data3, t3 = run_sim("step_U05", init_step_U05!, T_total)

println("\n"^2 * "="^70)
println("ALL RUNS COMPLETE — see step_sim_*.csv for full traces")
println("="^70)
