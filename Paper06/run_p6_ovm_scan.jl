#!/usr/bin/env julia
"""
P6 OVM-HR model: Optimal Velocity + Hard Rod with anticipatory delay.

Each rod adjusts velocity toward V_opt(gap(t-τ_r)) at rate α.
Hard-core elastic collision as safety net when gap reaches 0.
The delay is BEFORE the response (anticipatory), not after collision.

Quick scan over (α, τ_r) to find the shock-onset regime.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "Papers", "Paper01"))

include(joinpath(@__DIR__, "..", "Papers", "Paper01", "src", "event_driven", "curves.jl"))
include(joinpath(@__DIR__, "..", "Papers", "Paper01", "src", "event_driven", "arc_length_map.jl"))

using Statistics, Printf, Random

const A_SEMI  = 2.0
const KT      = 1.0
const NN      = 400
const ETA     = 0.10
const N_MODES = 7
const M_MAX   = 5.0
const U_0     = 0.5
const T_MAX   = 50.0
const DT_SAVE = 0.10
const DT_INT  = 5e-4
const N_SEEDS = 20

function elastic_collision_pair(v_i, v_j, m_i, m_j)
    M = m_i + m_j
    v_i_new = ((m_i - m_j)*v_i + 2*m_j*v_j) / M
    v_j_new = (2*m_i*v_i + (m_j - m_i)*v_j) / M
    return v_i_new, v_j_new
end

function step_ic(N, L, U_0, kT, masses, a_rod, rng)
    s  = [(i - 0.5) * L / N for i in 1:N]
    vs = [randn(rng) * sqrt(kT/masses[i]) for i in 1:N]
    for i in 1:N
        vs[i] += cos(2π * s[i] / L) > 0 ? U_0 : -U_0
    end
    p_tot = sum(masses .* vs)
    vs .-= p_tot / sum(masses)
    return s, vs
end

function compute_gaps!(g, s, N, L, a_rod)
    @inbounds for i in 1:N
        j = mod1(i+1, N)
        g[i] = (j > i ? s[j] - s[i] : s[j] - s[i] + L) - a_rod
    end
end

function run_ovm(s0, vs0, masses, L, T_max, dt_save, a_rod,
                 τ_r, α, v_free, g_c; dt=DT_INT, n_max=N_MODES)
    N_ = length(s0)
    s  = copy(s0)
    vs = copy(vs0)

    n_hist = max(1, Int(ceil(τ_r / dt)))
    gap_buf = zeros(N_, n_hist)
    buf_write = 1

    g_now = zeros(N_)
    compute_gaps!(g_now, s, N_, L, a_rod)
    for j in 1:n_hist
        gap_buf[:, j] .= g_now
    end

    save_times = collect(0.0:dt_save:T_max)
    T_steps = length(save_times)
    f_cos = zeros(n_max, T_steps)
    f_sin = zeros(n_max, T_steps)
    save_idx = 1
    t_now = 0.0
    n_colls = 0

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
        while save_idx <= T_steps && save_times[save_idx] <= t_now + dt/2
            take_obs!(save_idx)
            save_idx += 1
        end

        # Read delayed gaps (oldest entry, about to be overwritten)
        @inbounds for i in 1:N_
            g_del = gap_buf[i, buf_write]
            v_des = v_free * tanh(max(g_del, 0.0) / g_c)
            vs[i] += α * (v_des - vs[i]) * dt
        end

        s .+= vs .* dt
        t_now += dt

        compute_gaps!(g_now, s, N_, L, a_rod)
        gap_buf[:, buf_write] .= g_now
        buf_write = mod1(buf_write + 1, n_hist)

        # Resolve overlaps: elastic collision + push to contact
        for pass in 1:3
            any_overlap = false
            @inbounds for i in 1:N_
                if g_now[i] < 0.0
                    j = mod1(i+1, N_)
                    if vs[i] > vs[j]
                        vi_new, vj_new = elastic_collision_pair(
                            vs[i], vs[j], masses[i], masses[j])
                        vs[i] = vi_new; vs[j] = vj_new
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

    return f_cos, f_sin, save_times, n_colls
end

function main()
    curve = EllipsePolar(A_SEMI, A_SEMI)
    L = build_arc_length_map(curve).L
    a_rod = ETA * L / NN
    g_mean = L / NN - a_rod
    @printf("L=%.4f  a_rod=%.6f  g_mean=%.5f\n\n", L, a_rod, g_mean)

    masses_template = [iseven(i) ? 1.0 : M_MAX for i in 1:NN]

    v_free = 1.0
    g_c_vals  = [g_mean/2, g_mean, 2*g_mean]
    α_vals    = [0.1, 0.3, 0.5, 1.0]
    τ_vals    = [0.0, 1.0, 2.0, 3.0, 4.0]

    println("─── Quick scan: α × g_c × τ_r  ($(N_SEEDS) seeds each) ───\n")

    for g_c in g_c_vals
        @printf("g_c = %.5f (g_mean/%.1f)\n", g_c, g_mean/g_c)
        @printf("%6s", "α\\τ_r")
        for τ_r in τ_vals
            @printf("  %6.1f", τ_r)
        end
        println()

        for α in α_vals
            @printf("%6.2f", α)
            for τ_r in τ_vals
                R2_sum = 0.0
                for seed in 1:N_SEEDS
                    rng = MersenneTwister(seed*1009 + 77777)
                    masses = shuffle(rng, copy(masses_template))
                    s0, vs0 = step_ic(NN, L, U_0, KT, masses, a_rod, rng)
                    fc, fs, st, nc = run_ovm(s0, vs0, masses, L, T_MAX,
                        DT_SAVE, a_rod, τ_r, α, v_free, g_c)
                    rho_n = sqrt.(fc.^2 .+ fs.^2)
                    tp = argmax(rho_n[1, :])
                    R2_sum += rho_n[2, tp] / max(rho_n[1, tp], 1e-15)
                end
                @printf("  %6.3f", R2_sum / N_SEEDS)
            end
            println()
        end
        println()
    end
end

main()
