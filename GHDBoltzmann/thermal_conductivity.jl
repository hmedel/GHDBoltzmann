#=
Thermal conductivity λ_T(m_max) via Chapman-Enskog for the
GHD-Boltzmann binary mixture (Paper V Sec V extension).

Setup: same 3-pt Lagrange + Bobylev-Palczewski-Schneider matrix L̃
acting on h = (h_L, h_H). For a pure temperature gradient
(no density or velocity gradient at the slow-field level), the
source is

  s_α(v) = (v/T)·(m_α v²/(2T) - C)·(∂_s T)

where C is fixed by the orthogonality condition against the
total-momentum invariant ψ_3 = m_α v: in 1D ideal gas C = 3/2
(the dimensionless ``Eucken constant'' for 1D).

Heat flux moment:
  q = ∫ v·(m_α v²/2 - 3T/2)·ρ_α^(0)·h_α^(1) dv  (summed over α)
λ_T = -q / ∂_s T  (with ∂_s T = 1 normalisation).

Connection to sound damping (1D ideal binary):
  γ_sound = (2/(3 m̄ n_tot T)) · λ_T · k²
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")  # reuse build_L_matrix_3pt, locate3, lagrange3_weights
using LinearAlgebra, Printf

println()
println("="^70)
println("Thermal conductivity λ_T via Chapman-Enskog")
println("="^70)

# Verify orthogonality condition: derive C from
# ⟨ψ_3, v·(m_α v²/(2T) - C)·ρ^(0)⟩_{summed over α} = 0
# Result: C = 3/2 in 1D (computed analytically).
function compute_lambda_T(m_max::Float64; n_each=rho_tot/2, kT=1.0,
                           N_v=N_v, V_max=V_max, sigma=sigma_val)
    dv = 2*V_max/N_v
    v_grid = [-V_max + (j-0.5)*dv for j in 1:N_v]
    Z_L = sqrt(2π*kT); Z_H = sqrt(2π*kT/m_max)
    f_L = [(n_each/Z_L)*exp(-v_grid[j]^2/(2*kT)) for j in 1:N_v]
    f_H = [(n_each/Z_H)*exp(-m_max*v_grid[j]^2/(2*kT)) for j in 1:N_v]

    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma, 1.0, m_max, v_grid, f_L, f_H)

    # Thermal source (h-form, ∂_s T = 1):
    # s_α(v) = (v/T)·(m_α v²/(2T) - 3/2)
    b = zeros(2*N_v)
    for j in 1:N_v
        v = v_grid[j]
        b[j]         = (v/kT) * (1.0*v^2/(2*kT)    - 1.5)
        b[N_v + j]   = (v/kT) * (m_max*v^2/(2*kT)  - 1.5)
    end

    # Check source orthogonality to invariants (sanity)
    mom_overlap = sum(1.0*v_grid[j]*f_L[j]*b[j]    for j in 1:N_v)*dv +
                  sum(m_max*v_grid[j]*f_H[j]*b[N_v+j] for j in 1:N_v)*dv

    # Solve L̃ h = b via symmetric eigendecomp in f^eq metric
    d = vcat([sqrt(max(f_L[j], 1e-300)) for j in 1:N_v],
             [sqrt(max(f_H[j], 1e-300)) for j in 1:N_v])
    D_inv = 1.0 ./ d
    S_mat = (d .* L̃) .* D_inv'
    S_sym = (S_mat + S_mat')/2
    ev = eigen(S_sym)
    λ = ev.values; Q = ev.vectors
    B = d .* b

    λ_max = maximum(abs.(λ))
    keep = abs.(λ) .> 1e-6 * λ_max  # canonical threshold from audit
    λ_inv = [keep[i] ? 1.0/λ[i] : 0.0 for i in 1:length(λ)]
    h = D_inv .* (Q * (λ_inv .* (Q' * B)))

    # Heat flux: q = ∫ v·(m_α v²/2 - 3T/2)·ρ_α^(0)·h_α dv
    q = 0.0
    for j in 1:N_v
        v = v_grid[j]
        kL = v * (1.0*v^2/2 - 1.5*kT) * f_L[j] * h[j]
        kH = v * (m_max*v^2/2 - 1.5*kT) * f_H[j] * h[N_v + j]
        q += (kL + kH) * dv
    end
    λ_T = -q  # since ∂_s T = 1

    return λ_T, mom_overlap, count(keep)
end

# Sound-damping prediction in 1D binary:
# γ_sound = (2/(3 m̄ n_tot T)) · λ_T · k²
function gamma_from_lambdaT(λ_T, m_max; n_tot=rho_tot, T=1.0, k=2π/L_arc)
    m̄ = (1.0 + m_max)/2
    return (2.0/(3*m̄*n_tot*T)) * λ_T * k^2
end

println()
println("Sound-damping observed in DVM (Sec VI long sim, m_max=5): γ ≈ 0.042")
println("Empirical MD damping (P2 ref, m_max=5): γ ≈ 0.02 - 0.03")
println()
println("| m_max | λ_T | mom_overlap | n_keep | γ_pred = (2/3m̄nT)λk² | ratio vs 0.042 |")
println("|---|---|---|---|---|---|")
for m_max in (2.0, 5.0, 10.0, 20.0, 100.0)
    λ_T, mom, nk = compute_lambda_T(m_max)
    γ_pred = gamma_from_lambdaT(λ_T, m_max)
    @printf("| %g | %.4f | %.2e | %d | %.5f | %.3f |\n",
            m_max, λ_T, mom, nk, γ_pred, γ_pred/0.042)
end
