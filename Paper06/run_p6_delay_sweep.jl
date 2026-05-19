#!/usr/bin/env julia
"""
Paper VI numerical test: shock-onset threshold τ_r^c with reaction-time delay.

Model B (contact-freeze): at collision t_c, both rods instantly adopt
v_min = min(v_i, v_j) and remain in contact for τ_r; at t_c + τ_r the
full elastic velocity exchange is applied. Preserves cyclic-order
invariant (rods cannot pass through one another during freeze).

Sweep: τ_r ∈ {0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0} at fixed (m_max=5, U_0=0.5)
Diagnostic: R_2 = |ρ̂_2|/|ρ̂_1| at first peak of |ρ̂_1| — see Paper VI Sec VI.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "Papers", "Paper01"))

include(joinpath(@__DIR__, "..", "Papers", "Paper01", "src", "event_driven", "curves.jl"))
include(joinpath(@__DIR__, "..", "Papers", "Paper01", "src", "event_driven", "arc_length_map.jl"))

using HDF5, Statistics, Printf, Random, Dates

const A_SEMI  = 2.0
const KT      = 1.0
const N       = 400
const ETA     = 0.10
const T_MAX   = 50.0
const DT_SAVE = 0.10
const N_MODES = 7

const M_MAX   = 5.0
const U_0     = 0.5
const TAU_R_GRID = [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
const N_SEEDS = 200

const OUT_DIR = let
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    joinpath(@__DIR__, "results", "p6_delay_$(stamp)")
end
mkpath(OUT_DIR)

# ─── Physics helpers ─────────────────────────────────────────────────────────

function gap_fn(s, i, N, L, a_rod)
    j  = mod1(i+1, N)
    Δs = j > i ? s[j] - s[i] : s[j] - s[i] + L
    Δs - a_rod
end

function coll_dt(s, vs, i, N, L, a_rod)
    j  = mod1(i+1, N)
    dv = vs[j] - vs[i]
    dv >= 0.0 && return Inf
    g = gap_fn(s, i, N, L, a_rod)
    g <= 0.0 && return 0.0
    g / (-dv)
end

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

# ─── Main simulation: Model B contact-freeze ─────────────────────────────────

function run_with_delay(s0, vs0, masses, L, T_max, dt_save, a_rod, τ_r;
                        n_max=N_MODES)
    N_ = length(s0)
    s  = copy(s0)
    vs = copy(vs0)
    t_now = 0.0

    # Model B per-particle state
    t_release = fill(-1.0, N_)
    v_pending = fill(NaN, N_)
    # Per-pair state: pair_frozen[i] = true ↔ pair (i, mod1(i+1,N_)) in contact
    pair_frozen = falses(N_)

    t_coll = [coll_dt(s, vs, i, N_, L, a_rod) for i in 1:N_]

    save_times = collect(0.0:dt_save:T_max)
    T_steps    = length(save_times)
    f_cos = zeros(n_max, T_steps)
    f_sin = zeros(n_max, T_steps)
    save_idx = 1

    function take_obs(idx, t_snap)
        Δ = t_snap - t_now
        @inbounds for n in 1:n_max
            ac = 0.0; as = 0.0
            kn = 2π * n / L
            for k in 1:N_
                phase = kn * mod(s[k] + vs[k]*Δ, L)
                ac += cos(phase); as += sin(phase)
            end
            f_cos[n, idx] = ac / N_
            f_sin[n, idx] = as / N_
        end
    end

    function next_release_time()
        t_min = Inf
        @inbounds for k in 1:N_
            if t_release[k] >= 0.0 && t_release[k] < t_min
                t_min = t_release[k]
            end
        end
        return t_min
    end

    function recompute_pair!(k)
        t_coll[k] = pair_frozen[k] ? Inf : t_now + coll_dt(s, vs, k, N_, L, a_rod)
    end

    iter = 0
    iter_max = 200_000_000

    while iter < iter_max
        iter += 1
        t_next_save = save_idx <= T_steps ? save_times[save_idx] : Inf

        i_coll = argmin(t_coll)
        t_event_coll = t_coll[i_coll]
        t_event_rel  = next_release_time()

        kinds = (:save, :release, :coll)
        ts    = (t_next_save, t_event_rel, t_event_coll)
        k_idx = argmin(ts)
        kind  = kinds[k_idx]
        t_event = ts[k_idx]

        if t_event > T_max
            if save_idx <= T_steps && save_times[save_idx] <= T_max + 1e-12
                take_obs(save_idx, save_times[save_idx])
                save_idx += 1
            end
            break
        end

        if kind == :save
            take_obs(save_idx, t_next_save)
            save_idx += 1
            continue
        end

        # Advance all positions to t_event
        Δt = t_event - t_now
        s .+= vs .* Δt
        t_now = t_event

        if kind == :release
            # ── RELEASE: apply v_pending to ALL particles releasing now ──
            tol = 1e-12
            @inbounds for k in 1:N_
                if t_release[k] >= 0.0 && t_release[k] - t_now < tol
                    vs[k] = v_pending[k]
                    t_release[k] = -1.0
                    v_pending[k] = NaN
                end
            end
            # Unfreeze pairs where both particles have released
            @inbounds for k in 1:N_
                if pair_frozen[k]
                    j = mod1(k+1, N_)
                    if t_release[k] < 0.0 && t_release[j] < 0.0
                        pair_frozen[k] = false
                    end
                end
            end
            # Recompute collision times for all pairs touching a released rod
            @inbounds for k in 1:N_
                recompute_pair!(k)
            end
        else
            # ── COLLISION: pair (i_coll, j_coll) ────────────────────────
            j_coll = mod1(i_coll + 1, N_)

            v_i_new, v_j_new = elastic_collision_pair(
                vs[i_coll], vs[j_coll], masses[i_coll], masses[j_coll])

            if τ_r > 0.0
                # Contact-freeze: both rods adopt v_min immediately
                v_min = min(vs[i_coll], vs[j_coll])
                vs[i_coll] = v_min
                vs[j_coll] = v_min
                # Queue elastic exchange at t_now + τ_r
                v_pending[i_coll] = v_i_new
                v_pending[j_coll] = v_j_new
                t_release[i_coll] = t_now + τ_r
                t_release[j_coll] = t_now + τ_r
                # Freeze this pair
                pair_frozen[i_coll] = true
                t_coll[i_coll] = Inf
                # Recompute neighboring pairs (i_coll velocity changed)
                recompute_pair!(mod1(i_coll - 1, N_))
                recompute_pair!(j_coll)
            else
                # τ_r = 0: instantaneous elastic swap (P3 baseline)
                vs[i_coll] = v_i_new
                vs[j_coll] = v_j_new
                for k in (mod1(i_coll-1, N_), i_coll, j_coll, mod1(j_coll+1, N_))
                    t_coll[k] = t_now + coll_dt(s, vs, k, N_, L, a_rod)
                end
            end
        end
    end

    iter >= iter_max && @warn "iter_max hit at t=$t_now"
    return f_cos, f_sin, save_times
end

# ─── Driver ──────────────────────────────────────────────────────────────────

function main()
    println("P6 delay sweep (Model B contact-freeze): τ_r ∈ ", TAU_R_GRID)
    curve = EllipsePolar(A_SEMI, A_SEMI)
    L = build_arc_length_map(curve).L
    a_rod = ETA * L / N
    v_th = sqrt(KT)
    @printf("L=%.4f  a_rod=%.6f  v_th=%.4f\n\n", L, a_rod, v_th)

    masses_template = [iseven(i) ? 1.0 : M_MAX for i in 1:N]
    save_times = collect(0.0:DT_SAVE:T_MAX)
    T_steps = length(save_times)

    for τ_r in TAU_R_GRID
        @printf("─── τ_r = %.2f ───\n", τ_r)
        f_cos_sum = zeros(N_MODES, T_steps)
        f_sin_sum = zeros(N_MODES, T_steps)
        f_cos_sq  = zeros(N_MODES, T_steps)
        f_sin_sq  = zeros(N_MODES, T_steps)
        lock_obj = ReentrantLock()

        t_wall = @elapsed Threads.@threads for seed in 1:N_SEEDS
            rng = MersenneTwister(seed*1009 + Int(round(τ_r*1000))*7919 + 22222)
            masses = shuffle(rng, copy(masses_template))
            s0, vs0 = step_ic(N, L, U_0, KT, masses, a_rod, rng)
            f_c, f_s, _ = run_with_delay(s0, vs0, masses, L, T_MAX, DT_SAVE, a_rod, τ_r)
            lock(lock_obj) do
                f_cos_sum .+= f_c; f_sin_sum .+= f_s
                f_cos_sq  .+= f_c.^2; f_sin_sq .+= f_s.^2
            end
        end

        f_cos_mean = f_cos_sum ./ N_SEEDS
        f_sin_mean = f_sin_sum ./ N_SEEDS
        f_cos_var  = f_cos_sq ./ N_SEEDS .- f_cos_mean.^2
        f_sin_var  = f_sin_sq ./ N_SEEDS .- f_sin_mean.^2

        rho_n = sqrt.(f_cos_mean.^2 .+ f_sin_mean.^2)

        rho_1 = rho_n[1, :]
        t_peak_idx = argmax(rho_1)
        R_2 = rho_n[2, t_peak_idx] / max(rho_n[1, t_peak_idx], 1e-15)

        @printf("  %d seeds in %.1fs  →  t_peak=%.2f  |ρ̂_1|=%.4f  R_2=%.4f\n",
                N_SEEDS, t_wall, save_times[t_peak_idx], rho_1[t_peak_idx], R_2)

        h5open(joinpath(OUT_DIR, @sprintf("modes_tau%.2f.h5", τ_r)), "w") do fh
            write(fh, "f_cos_mean", f_cos_mean)
            write(fh, "f_sin_mean", f_sin_mean)
            write(fh, "f_cos_var",  f_cos_var)
            write(fh, "f_sin_var",  f_sin_var)
            write(fh, "rho_n",      rho_n)
            write(fh, "times",      save_times)
            write(fh, "tau_r",      τ_r)
            write(fh, "t_peak_idx", t_peak_idx)
            write(fh, "R_2",        R_2)
        end
    end

    println("\nResults in: ", OUT_DIR)
end

main()
