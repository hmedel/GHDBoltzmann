#!/usr/bin/env julia
"""
P6 OVM-HR scan 2: weak coupling α ∈ {0.05,0.08,0.12,0.15}, T_MAX=135.
Diagnostics: R₂ at t_peak, max|ρ̂₂| for t>20, max|ρ̂₂| for t>50.
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
const T_MAX   = 135.0
const DT_SAVE = 0.10
const DT_INT  = 5e-4
const N_SEEDS = 30

function elastic_collision_pair(v_i, v_j, m_i, m_j)
    M = m_i + m_j
    ((m_i - m_j)*v_i + 2*m_j*v_j) / M,
    (2*m_i*v_i + (m_j - m_i)*v_j) / M
end

function step_ic(N, L, U_0, kT, masses, a_rod, rng)
    s  = [(i - 0.5) * L / N for i in 1:N]
    vs = [randn(rng) * sqrt(kT/masses[i]) for i in 1:N]
    for i in 1:N
        vs[i] += cos(2π * s[i] / L) > 0 ? U_0 : -U_0
    end
    vs .-= sum(masses .* vs) / sum(masses)
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
    s  = copy(s0); vs = copy(vs0)

    n_hist = max(1, Int(ceil(τ_r / dt)))
    gap_buf = zeros(N_, n_hist)
    buf_w = 1

    g_now = zeros(N_)
    compute_gaps!(g_now, s, N_, L, a_rod)
    for j in 1:n_hist; gap_buf[:, j] .= g_now; end

    save_times = collect(0.0:dt_save:T_max)
    T_steps = length(save_times)
    f_cos = zeros(n_max, T_steps)
    f_sin = zeros(n_max, T_steps)
    save_idx = 1; t_now = 0.0

    function take_obs!(idx)
        @inbounds for n in 1:n_max
            ac = 0.0; as = 0.0; kn = 2π * n / L
            for k in 1:N_
                ph = kn * mod(s[k], L)
                ac += cos(ph); as += sin(ph)
            end
            f_cos[n, idx] = ac / N_; f_sin[n, idx] = as / N_
        end
    end

    for step in 1:Int(ceil(T_max / dt))
        while save_idx <= T_steps && save_times[save_idx] <= t_now + dt/2
            take_obs!(save_idx); save_idx += 1
        end

        @inbounds for i in 1:N_
            g_del = gap_buf[i, buf_w]
            v_des = v_free * tanh(max(g_del, 0.0) / g_c)
            vs[i] += α * (v_des - vs[i]) * dt
        end

        s .+= vs .* dt; t_now += dt

        compute_gaps!(g_now, s, N_, L, a_rod)
        gap_buf[:, buf_w] .= g_now
        buf_w = mod1(buf_w + 1, n_hist)

        for pass in 1:3
            any_ov = false
            @inbounds for i in 1:N_
                if g_now[i] < 0.0
                    j = mod1(i+1, N_)
                    if vs[i] > vs[j]
                        vs[i], vs[j] = elastic_collision_pair(
                            vs[i], vs[j], masses[i], masses[j])
                    end
                    ov = -g_now[i]; M = masses[i] + masses[j]
                    s[i] -= ov * masses[j] / M
                    s[j] += ov * masses[i] / M
                    any_ov = true
                end
            end
            any_ov || break
            compute_gaps!(g_now, s, N_, L, a_rod)
        end
    end
    while save_idx <= T_steps; take_obs!(save_idx); save_idx += 1; end
    return f_cos, f_sin, save_times
end

function main()
    curve = EllipsePolar(A_SEMI, A_SEMI)
    L = build_arc_length_map(curve).L
    a_rod = ETA * L / NN
    g_mean = L / NN - a_rod
    @printf("L=%.4f  a_rod=%.6f  g_mean=%.5f  T_MAX=%.0f  seeds=%d\n\n",
            L, a_rod, g_mean, T_MAX, N_SEEDS)

    masses_template = [iseven(i) ? 1.0 : M_MAX for i in 1:NN]
    v_free = 1.0
    g_c = g_mean

    α_vals = [0.05, 0.08, 0.12, 0.15]
    τ_vals = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    save_times = collect(0.0:DT_SAVE:T_MAX)
    i20 = findfirst(save_times .>= 20.0)
    i50 = findfirst(save_times .>= 50.0)

    for α in α_vals
        @printf("═══ α = %.2f,  g_c = %.5f ═══\n", α, g_c)
        @printf("%6s  %7s  %7s  %7s  %7s  %7s\n",
                "τ_r", "R₂@pk", "|ρ̂₂|pk", "|ρ̂₂|>20", "|ρ̂₂|>50", "|ρ̂₁|pk")

        for τ_r in τ_vals
            fc_sum = zeros(N_MODES, length(save_times))
            fs_sum = zeros(N_MODES, length(save_times))

            t_wall = @elapsed for seed in 1:N_SEEDS
                rng = MersenneTwister(seed*1009 + 77777)
                masses = shuffle(rng, copy(masses_template))
                s0, vs0 = step_ic(NN, L, U_0, KT, masses, a_rod, rng)
                fc, fs, _ = run_ovm(s0, vs0, masses, L, T_MAX,
                    DT_SAVE, a_rod, τ_r, α, v_free, g_c)
                fc_sum .+= fc; fs_sum .+= fs
            end

            fc_m = fc_sum ./ N_SEEDS; fs_m = fs_sum ./ N_SEEDS
            rho_n = sqrt.(fc_m.^2 .+ fs_m.^2)

            tp = argmax(rho_n[1, :])
            R2_pk = rho_n[2, tp] / max(rho_n[1, tp], 1e-15)
            rho2_pk = rho_n[2, tp]
            rho2_max20 = maximum(rho_n[2, i20:end])
            rho2_max50 = maximum(rho_n[2, i50:end])
            rho1_pk = rho_n[1, tp]

            @printf("%6.1f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  (%.0fs)\n",
                    τ_r, R2_pk, rho2_pk, rho2_max20, rho2_max50, rho1_pk, t_wall)
        end
        println()
    end
end

main()
