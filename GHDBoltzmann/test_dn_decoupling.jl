#=
Test: does the Δn mode decouple from the sound mode at the Euler order?

IC 1: pure Δn perturbation
  n_L(s) = (ρ/2)(1 + A cos(k₁s))
  n_H(s) = (ρ/2)(1 - A cos(k₁s))
  u(s)   = 0
  T      = 1
  → n_tot uniform, only Δn perturbed.

Predictions if my analysis is correct:
  - n_tot stays uniform (small numerical noise only)
  - u stays ~0 (no sound excitation)
  - Δn(t) decays as exp(-D_LH · k₁² · t) with D_LH ≈ 4 (Sec V)
    → decay rate ≈ 4 · 0.25 = 1.0 per unit time
    → half-life ≈ 0.7

If interdiffusion DID damp sound, we'd see sound mode excited even from
this pure-Δn IC. The test is decisive.

IC 2 (control): pure thermal perturbation
  T(s) = 1 + B cos(k₁s)
  n_L = n_H = ρ/2 uniform, u = 0
  → adiabatic coupling expected → sound mode excited.
=#
push!(LOAD_PATH, @__DIR__)
using GHDBoltzmannSolver
using Printf, DelimitedFiles

const L_arc = 12.57
const σ = 0.00314
const ρ = 400/L_arc
const k1 = 2π/L_arc
const A = 0.05
const m_max = 5.0

function fourier_mode(st, field_per_cell, k, N_s)
    a = 2.0*sum(field_per_cell .* cos.(k*st.s))/N_s
    b = 2.0*sum(field_per_cell .* sin.(k*st.s))/N_s
    return sqrt(a^2 + b^2)
end

function diagnostics(st)
    p = st.p
    n_L = vec(sum(st.rho_L, dims=2)) .* st.dv
    n_H = vec(sum(st.rho_H, dims=2)) .* st.dv
    n_tot = n_L .+ n_H
    Δn = (n_L .- n_H) ./ 2
    # momentum density / mass density = ū
    sum_pL = vec(st.rho_L * st.v) .* st.dv
    sum_pH = vec(st.rho_H * st.v) .* st.dv
    p_dens = p.m_L .* sum_pL .+ p.m_H .* sum_pH
    m_dens = p.m_L .* n_L .+ p.m_H .* n_H
    ū = p_dens ./ m_dens

    # Fourier mode 1 of each
    f_ntot = fourier_mode(st, n_tot ./ ρ .- 1, k1, p.N_s)
    f_Δn   = fourier_mode(st, Δn ./ ρ, k1, p.N_s)
    f_u    = fourier_mode(st, ū, k1, p.N_s)
    return f_ntot, f_Δn, f_u
end

# --- IC 1: pure Δn ---
println("="^70)
println("Test: pure Δn IC — should excite Δn mode only (no sound)")
println("="^70)
p = GHDBParams(L=L_arc, sigma=σ, m_L=1.0, m_H=m_max, kT=1.0,
               N_s=80, N_v=128, V_max=6.0, cfl=0.25)
st = GHDBState(p)
n_L_arr = [(ρ/2)*(1 + A*cos(k1*st.s[k])) for k in 1:p.N_s]
n_H_arr = [(ρ/2)*(1 - A*cos(k1*st.s[k])) for k in 1:p.N_s]
u_arr   = zeros(p.N_s)
init_local_maxwell!(st, n_L_arr, n_H_arr, u_arr, 1.0)

f0 = diagnostics(st)
@printf("  Initial: |n̂_tot|=%.4e  |Δn̂|=%.4e  |û|=%.4e\n", f0...)

# Run 4 time units (~ 4 e-foldings if D_LH·k²=1)
T_end = 4.0
records = Tuple{Float64, NTuple{3, Float64}}[]
cb(s) = diagnostics(s)
recs, _, _ = run_simulation!(st, T_end, save_every=0.05, callback=cb)

println("\nTime | |n̂_tot| | |Δn̂| | |û| | Δn̂_normalized")
for r in recs[1:5:end]
    t, (fnt, fdn, fu) = r
    @printf("  %.2f | %.4e | %.4e | %.4e | %.4f\n",
            t, fnt, fdn, fu, fdn/f0[2])
end

# Fit Δn decay
ts = [r[1] for r in recs]
Δns = [r[2][2] for r in recs]
log_Δn = log.(Δns)
# Linear fit log Δn = a + b·t  → decay rate = -b
mask = ts .> 0.2  # avoid IC transient
t_f = ts[mask]
l_f = log_Δn[mask]
n_pts = length(t_f)
t_bar = sum(t_f)/n_pts
l_bar = sum(l_f)/n_pts
b_slope = sum((t_f .- t_bar) .* (l_f .- l_bar)) / sum((t_f .- t_bar).^2)
@printf("\nFitted Δn decay rate: -b = %.4f  (predicted D_LH·k₁² = %.4f)\n",
        -b_slope, 4.04 * k1^2)
@printf("Ratio observed/predicted: %.3f\n", -b_slope / (4.04 * k1^2))
@printf("Final |n̂_tot|/|n̂_tot|_init = %.4e (should be ≪ 1 if decoupled)\n",
        recs[end][2][1]/f0[1])
@printf("Final |û|/(c_s·A) = %.4e (should be ≪ 1 if no sound excited)\n",
        recs[end][2][3]/(A*sqrt(3/3.0)/0.9))

# --- IC 2 (control): thermal-only ---
println("\n"*"="^70)
println("Control: thermal-only IC — should excite sound mode")
println("="^70)
st2 = GHDBState(p)
T_arr = [1.0 + A*cos(k1*st2.s[k]) for k in 1:p.N_s]
n_L_uniform = fill(ρ/2, p.N_s)
n_H_uniform = fill(ρ/2, p.N_s)
init_local_maxwell!(st2, n_L_uniform, n_H_uniform, zeros(p.N_s), T_arr)

f0_2 = diagnostics(st2)
@printf("  Initial: |n̂_tot|=%.4e  |Δn̂|=%.4e  |û|=%.4e\n", f0_2...)
recs2, _, _ = run_simulation!(st2, T_end, save_every=0.05, callback=cb)
println("\nTime | |n̂_tot| | |Δn̂| | |û|")
for r in recs2[1:5:end]
    t, (fnt, fdn, fu) = r
    @printf("  %.2f | %.4e | %.4e | %.4e\n", t, fnt, fdn, fu)
end
@printf("\nPeak |û| during evolution = %.4e (should be ≫ 0 if sound coupled)\n",
        maximum([r[2][3] for r in recs2]))
@printf("Peak |Δn̂| = %.4e (should remain near initial value)\n",
        maximum([r[2][2] for r in recs2]))
