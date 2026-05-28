#!/usr/bin/env julia
"""
Paper VII: breaking integrability in the hard-rod gas.

Three channels:
  (a) ε > 0  — inelastic collisions (r = 1 - ε)
  (b) α > 0  — non-reciprocal OVM: dv_i/dt = α[V_opt(g_i^fwd(t-τ)) - v_i]
  (c) τ_r > 0 — reaction-time delay on gap sensing

Time-stepping engine with elastic/inelastic hard-core overlap resolution.
Diagnostic: R_2 = |ρ̂_2| / |ρ̂_1| at the first peak of |ρ̂_1|.
"""

using HDF5, Statistics, Printf, Random, Dates

# ─── Parameters ──────────────────────────────────────────────────────────────

const R_RING  = 2.0
const L_RING  = 2π * R_RING
const KT      = 1.0
const NN      = 400
const ETA     = 0.10
const N_MODES = 7
const M_MAX   = 5.0
const U_0     = 0.5
const T_MAX   = parse(Float64, get(ENV, "P7_TMAX", "80.0"))
const DT_SAVE = 0.10
const DT_INT  = 5e-4
const N_SEEDS = parse(Int, get(ENV, "P7_NSEEDS", "40"))

const A_ROD   = ETA * L_RING / NN
const G_MEAN  = L_RING / NN - A_ROD

# OVM parameters
const V_FREE  = 1.0
const G_C     = G_MEAN

# ─── Scan modes ──────────────────────────────────────────────────────────────

const TAU_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

const SCAN_MODE = get(ENV, "P7_SCAN", "a")

const SCAN_A_EPS = parse.(Float64, split(get(ENV, "P7_EPS_GRID", "0.01,0.05,0.10,0.20"), ","))
const SCAN_B_ALPHA = parse.(Float64, split(get(ENV, "P7_ALPHA_GRID", "0.1,0.5,1.0,2.0"), ","))

# ─── Physics ─────────────────────────────────────────────────────────────────

function collision_pair(v_i, v_j, m_i, m_j, r)
    M = m_i + m_j
    v_i_new = ((m_i - r*m_j)*v_i + (1+r)*m_j*v_j) / M
    v_j_new = ((1+r)*m_i*v_i + (m_j - r*m_i)*v_j) / M
    return v_i_new, v_j_new
end

function step_ic(N, L, U_0, kT, masses, a_rod, rng)
    s  = [(i - 0.5) * L / N for i in 1:N]
    vs = [randn(rng) * sqrt(kT / masses[i]) for i in 1:N]
    for i in 1:N
        vs[i] += cos(2π * s[i] / L) > 0 ? U_0 : -U_0
    end
    p_tot = sum(masses .* vs)
    vs .-= p_tot / sum(masses)
    return s, vs
end

function compute_gaps!(g, s, N, L, a_rod)
    @inbounds for i in 1:N
        j = mod1(i + 1, N)
        g[i] = (j > i ? s[j] - s[i] : s[j] - s[i] + L) - a_rod
    end
end

# ─── Simulation engine ──────────────────────────────────────────────────────

function run_sim(s0, vs0, masses, L, T_max, dt_save, a_rod,
                 τ_r, α, ε;
                 dt=DT_INT, n_max=N_MODES, v_free=V_FREE, g_c=G_C)
    N_ = length(s0)
    r  = 1.0 - ε
    s  = copy(s0)
    vs = copy(vs0)

    # Delay buffer for gap sensing
    n_hist = max(1, Int(ceil(τ_r / dt)))
    gap_buf = zeros(N_, n_hist)
    buf_idx = 1

    g_now = zeros(N_)
    compute_gaps!(g_now, s, N_, L, a_rod)
    for j in 1:n_hist
        gap_buf[:, j] .= g_now
    end

    # Observables
    save_times = collect(0.0:dt_save:T_max)
    T_steps = length(save_times)
    f_cos = zeros(n_max, T_steps)
    f_sin = zeros(n_max, T_steps)
    save_idx = 1
    t_now = 0.0
    n_colls = 0
    KE_lost = 0.0

    function take_obs!(idx)
        @inbounds for n in 1:n_max
            ac = 0.0; as = 0.0
            kn = 2π * n / L
            for k in 1:N_
                phase = kn * mod(s[k], L)
                ac += cos(phase); as += sin(phase)
            end
            f_cos[n, idx] = ac / N_
            f_sin[n, idx] = as / N_
        end
    end

    n_steps = Int(ceil(T_max / dt))

    for step in 1:n_steps
        while save_idx <= T_steps && save_times[save_idx] <= t_now + dt / 2
            take_obs!(save_idx)
            save_idx += 1
        end

        # OVM force (non-reciprocal: each rod senses gap ahead)
        if α > 0.0
            @inbounds for i in 1:N_
                g_del = gap_buf[i, buf_idx]
                v_des = v_free * tanh(max(g_del, 0.0) / g_c)
                vs[i] += α * (v_des - vs[i]) * dt
            end
        end

        # Free streaming
        s .+= vs .* dt
        t_now += dt

        # Update gaps and delay buffer
        compute_gaps!(g_now, s, N_, L, a_rod)
        gap_buf[:, buf_idx] .= g_now
        buf_idx = mod1(buf_idx + 1, n_hist)

        # Resolve overlaps
        for pass in 1:3
            any_overlap = false
            @inbounds for i in 1:N_
                if g_now[i] < 0.0
                    j = mod1(i + 1, N_)
                    if vs[i] > vs[j]
                        KE_before = 0.5 * (masses[i]*vs[i]^2 + masses[j]*vs[j]^2)
                        vi_new, vj_new = collision_pair(vs[i], vs[j], masses[i], masses[j], r)
                        vs[i] = vi_new; vs[j] = vj_new
                        KE_after = 0.5 * (masses[i]*vs[i]^2 + masses[j]*vs[j]^2)
                        KE_lost += KE_before - KE_after
                        n_colls += 1
                    end
                    overlap = -g_now[i]
                    M = masses[i] + masses[j]
                    s[i] -= overlap * masses[j] / M
                    s[j] += overlap * masses[i] / M
                    any_overlap = true
                end
            end
            any_overlap || break
            compute_gaps!(g_now, s, N_, L, a_rod)
        end
    end

    while save_idx <= T_steps
        take_obs!(save_idx)
        save_idx += 1
    end

    return f_cos, f_sin, save_times, n_colls, KE_lost
end

# ─── Analysis ────────────────────────────────────────────────────────────────

const T_WIN_LO = parse(Float64, get(ENV, "P7_TWIN_LO", "2.0"))
const T_WIN_HI = parse(Float64, get(ENV, "P7_TWIN_HI", "8.0"))

function compute_R2(f_cos, f_sin, save_times)
    rho_n = sqrt.(f_cos.^2 .+ f_sin.^2)
    rho_1 = rho_n[1, :]

    # R₂ at global peak of |ρ̂₁|
    t_peak = argmax(rho_1)
    R2_peak = rho_n[2, t_peak] / max(rho_n[1, t_peak], 1e-15)

    # R₂ at peak of |ρ̂₁| within early window [T_WIN_LO, T_WIN_HI]
    win_mask = T_WIN_LO .<= save_times .<= T_WIN_HI
    if any(win_mask)
        win_idx = findall(win_mask)
        t_win_peak = win_idx[argmax(rho_1[win_idx])]
        R2_win = rho_n[2, t_win_peak] / max(rho_n[1, t_win_peak], 1e-15)
    else
        t_win_peak = t_peak
        R2_win = R2_peak
    end

    return R2_peak, R2_win, save_times[t_peak], save_times[t_win_peak], rho_1[t_peak]
end

# ─── Sweep driver ────────────────────────────────────────────────────────────

function run_sweep(label, ε, α, τ_grid, out_dir)
    masses_template = [iseven(i) ? 1.0 : M_MAX for i in 1:NN]
    save_times = collect(0.0:DT_SAVE:T_MAX)
    T_steps = length(save_times)

    @printf("─── %s: ε=%.3f  α=%.2f  [window %.1f–%.1f] ───\n", label, ε, α, T_WIN_LO, T_WIN_HI)
    @printf("%8s  %8s  %8s  %8s  %8s  %8s  %8s  %10s\n",
            "τ_r", "R₂peak", "R₂win", "±σ_win", "t_peak", "t_wp", "|ρ̂₁|", "colls/rod")

    for τ_r in τ_grid
        R2p_vals = zeros(N_SEEDS)
        R2w_vals = zeros(N_SEEDS)
        f_cos_sum = zeros(N_MODES, T_steps)
        f_sin_sum = zeros(N_MODES, T_steps)
        total_colls = 0
        total_KE = 0.0
        lock_obj = ReentrantLock()

        t_wall = @elapsed Threads.@threads for seed in 1:N_SEEDS
            rng = MersenneTwister(seed * 1009 + Int(round(τ_r * 1000)) * 7919 +
                                  Int(round(ε * 1e6)) * 13 + Int(round(α * 1e3)) * 41 + 55555)
            masses = shuffle(rng, copy(masses_template))
            s0, vs0 = step_ic(NN, L_RING, U_0, KT, masses, A_ROD, rng)
            fc, fs, _, nc, ke = run_sim(s0, vs0, masses, L_RING, T_MAX, DT_SAVE, A_ROD,
                                         τ_r, α, ε)
            R2p, R2w, _, _, _ = compute_R2(fc, fs, save_times)
            lock(lock_obj) do
                R2p_vals[seed] = R2p
                R2w_vals[seed] = R2w
                f_cos_sum .+= fc
                f_sin_sum .+= fs
                total_colls += nc
                total_KE += ke
            end
        end

        R2p_mean = mean(R2p_vals)
        R2w_mean = mean(R2w_vals)
        R2w_std  = std(R2w_vals) / sqrt(N_SEEDS)

        f_cos_mean = f_cos_sum ./ N_SEEDS
        f_sin_mean = f_sin_sum ./ N_SEEDS
        rho_n = sqrt.(f_cos_mean.^2 .+ f_sin_mean.^2)
        t_peak_idx = argmax(rho_n[1, :])

        win_mask = T_WIN_LO .<= save_times .<= T_WIN_HI
        win_idx = findall(win_mask)
        t_wp_idx = isempty(win_idx) ? t_peak_idx : win_idx[argmax(rho_n[1, win_idx])]

        @printf("%8.2f  %8.4f  %8.4f  %8.4f  %8.2f  %8.2f  %8.4f  %10.1f\n",
                τ_r, R2p_mean, R2w_mean, R2w_std,
                save_times[t_peak_idx], save_times[t_wp_idx],
                rho_n[1, t_peak_idx],
                total_colls / (N_SEEDS * NN))

        h5open(joinpath(out_dir, @sprintf("modes_eps%.3f_alpha%.2f_tau%.2f.h5", ε, α, τ_r)), "w") do fh
            write(fh, "f_cos_mean", f_cos_mean)
            write(fh, "f_sin_mean", f_sin_mean)
            write(fh, "rho_n", rho_n)
            write(fh, "times", save_times)
            write(fh, "tau_r", τ_r)
            write(fh, "epsilon", ε)
            write(fh, "alpha", α)
            write(fh, "R_2_peak_mean", R2p_mean)
            write(fh, "R_2_win_mean", R2w_mean)
            write(fh, "R_2_win_std", R2w_std)
            write(fh, "R_2_peak_all", R2p_vals)
            write(fh, "R_2_win_all", R2w_vals)
            write(fh, "t_peak_idx", t_peak_idx)
            write(fh, "t_win_peak_idx", t_wp_idx)
            write(fh, "n_colls", total_colls)
            write(fh, "KE_lost", total_KE)
            write(fh, "wall_time", t_wall)
        end
    end
    println()
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main()
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(@__DIR__, "results", "p7_scan_$(SCAN_MODE)_$(stamp)")
    mkpath(out_dir)

    @printf("Paper VII scan — L=%.4f  a=%.6f  g_mean=%.5f  N=%d  seeds=%d\n",
            L_RING, A_ROD, G_MEAN, NN, N_SEEDS)
    @printf("Mode: %s  |  T_max=%.1f  dt=%.1e\n\n", SCAN_MODE, T_MAX, DT_INT)

    if SCAN_MODE == "a"
        # Channel (a): inelasticity sweep, no OVM
        for ε in SCAN_A_EPS
            run_sweep("channel-a", ε, 0.0, TAU_GRID, out_dir)
        end

    elseif SCAN_MODE == "b"
        # Channel (b): OVM sweep, elastic collisions
        for α in SCAN_B_ALPHA
            run_sweep("channel-b", 0.0, α, TAU_GRID, out_dir)
        end

    elseif SCAN_MODE == "ab"
        # Combined (a+b): grid over (ε, α) × τ_r
        for ε in SCAN_A_EPS
            for α in SCAN_B_ALPHA
                run_sweep("combined", ε, α, TAU_GRID, out_dir)
            end
        end

    elseif SCAN_MODE == "baseline"
        # Integrable baseline: ε=0, α=0 (should reproduce P6 τ_r=0 result)
        run_sweep("baseline", 0.0, 0.0, TAU_GRID, out_dir)

    else
        error("Unknown P7_SCAN mode: $SCAN_MODE. Use: a, b, ab, or baseline")
    end

    println("Results saved to: ", out_dir)
end

main()
