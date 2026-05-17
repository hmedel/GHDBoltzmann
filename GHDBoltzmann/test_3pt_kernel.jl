#=
Regression tests for the 3-pt Lagrange port:
  (A) Mass, momentum, energy conservation under collision-only evolution
  (B) Sound speed validation: should give c_s_DVM/c_s_an closer to 1
      than the bilinear's 0.981
  (C) Comparison with bilinear baseline (numbers from prior runs)
=#
push!(LOAD_PATH, @__DIR__)
using GHDBoltzmannSolver
using Printf, Statistics

println("="^70)
println("3-pt Lagrange kernel: regression tests")
println("="^70)

# ---------- (A) Conservation under pure collision ----------
println("\n[A] Conservation under pure collision (uniform IC, N_s=1 effectively)")
println("    Expect: mass, momentum, energy preserved to machine precision")

p = GHDBParams(L=12.57, sigma=0.00314, m_L=1.0, m_H=5.0, kT=1.0,
               N_s=8, N_v=96, V_max=5.0, cfl=0.25)
st = GHDBState(p)
# Non-equilibrium: bimodal in velocity for both species
for k in 1:p.N_s
    for j in 1:p.N_v
        v_j = st.v[j]
        # L: shifted Maxwellian at u=+0.3
        st.rho_L[k, j] = exp(-0.5*p.m_L*(v_j - 0.3)^2)
        # H: shifted Maxwellian at u=-0.1
        st.rho_H[k, j] = 0.5 * exp(-0.5*p.m_H*(v_j + 0.1)^2)
    end
end

function global_moments(st)
    p = st.p
    M = 0.0; P = 0.0; E = 0.0
    for k in 1:p.N_s, j in 1:p.N_v
        v = st.v[j]
        M += (st.rho_L[k, j] + st.rho_H[k, j]) * st.ds * st.dv
        P += v * (p.m_L*st.rho_L[k, j] + p.m_H*st.rho_H[k, j]) * st.ds * st.dv
        E += 0.5 * v^2 * (p.m_L*st.rho_L[k, j] + p.m_H*st.rho_H[k, j]) * st.ds * st.dv
    end
    return M, P, E
end

M0, P0, E0 = global_moments(st)
@printf("  Initial: M=%.6e P=%.6e E=%.6e\n", M0, P0, E0)

# 1000 collision steps (no streaming)
dt = 0.01
for step in 1:1000
    collision_step_notime!(st, dt)
end
M1, P1, E1 = global_moments(st)
@printf("  After 1000 coll steps:\n")
@printf("    M=%.6e (ΔM/M = %.2e)\n", M1, (M1-M0)/abs(M0))
@printf("    P=%.6e (ΔP/|P| = %.2e)\n", P1, (P1-P0)/(abs(P0)+1e-30))
@printf("    E=%.6e (ΔE/E = %.2e)\n", E1, (E1-E0)/abs(E0))

# H-theorem check
S_final = kinetic_entropy(st)
@printf("  Final entropy S = %.6e\n", S_final)

# ---------- (B) Sound speed validation ----------
println("\n[B] Sound-speed validation at m_max ∈ {5, 10}")
println("    Bilinear baseline: c_s_DVM/c_s_an = 0.981 (constant)")
println("    Expect with 3-pt: ratio closer to 1.000")

const L_arc = 12.57
const sigma_val = 0.00314
const rho_tot = 400/L_arc
const A_rho = 0.05
const k1 = 2π/L_arc

function sound_test(m_max)
    p = GHDBParams(L=L_arc, sigma=sigma_val, m_L=1.0, m_H=m_max, kT=1.0,
                   N_s=80, N_v=128, V_max=6.0, cfl=0.25)
    st = GHDBState(p)
    m_bar = (1.0 + m_max)/2.0
    eta = sigma_val*rho_tot
    c_s_an = sqrt(3.0/m_bar)/(1.0 - eta)
    T_pred = 2π/(c_s_an*k1)

    n_L_p = [(rho_tot/2)*(1 + A_rho*cos(k1*st.s[k])) for k in 1:p.N_s]
    n_H_p = [(rho_tot/2)*(1 + A_rho*cos(k1*st.s[k])) for k in 1:p.N_s]
    u_p   = [A_rho*c_s_an*cos(k1*st.s[k]) for k in 1:p.N_s]
    init_local_maxwell!(st, n_L_p, n_H_p, u_p, 1.0)

    fourier(s) = begin
        nL = vec(sum(s.rho_L, dims=2))*s.dv
        nH = vec(sum(s.rho_H, dims=2))*s.dv
        delta = (nL .+ nH) ./ rho_tot .- 1.0
        a = 2.0*sum(delta .* cos.(2π*s.s/s.p.L))/s.p.N_s
        b = 2.0*sum(delta .* sin.(2π*s.s/s.p.L))/s.p.N_s
        return (a, b)
    end

    t_wall = @elapsed records, _, _ = run_simulation!(st, T_pred,
        save_every=T_pred/400, callback=fourier)
    times = [r[1] for r in records]
    as = [r[2][1] for r in records]; bs = [r[2][2] for r in records]
    thetas = [atan(b, a) for (a, b) in zip(as, bs)]
    cutoff = something(findfirst(t -> t > 0.05*T_pred, times), 5)
    cutoff = max(min(cutoff, length(times)), 4)
    t_fit = times[1:cutoff]; θ_fit = thetas[1:cutoff]
    ω_obs = sum(t_fit .* θ_fit)/sum(t_fit.^2)
    c_s_obs = ω_obs / k1
    return c_s_obs, c_s_an, c_s_obs/c_s_an, t_wall
end

println("\n| m_max | c_s_an | c_s_DVM (3pt) | ratio | wall (s) |")
println("|---|---|---|---|---|")
for m in (5.0, 10.0)
    c_obs, c_an, ratio, t_w = sound_test(m)
    @printf("| %g | %.4f | %.4f | %.4f | %.1f |\n", m, c_an, c_obs, ratio, t_w)
end
