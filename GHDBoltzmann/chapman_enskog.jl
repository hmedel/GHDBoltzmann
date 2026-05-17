#=
Chapman-Enskog interdiffusion coefficient D_LH(m_max) for the
GHD-Boltzmann binary mixture (Paper V Sec V).

CONSERVATIVE 3-POINT LAGRANGE INTERPOLATION VERSION.

The bilinear (2-point) interpolation of the original solver
preserves polynomials of degree 1, so collision invariants
{1, m_α v} are exact null modes but {m_α v²} is null only to
O(Δv²). For the CE extraction this O(Δv²) error appears as
three small-but-nonzero singular values of the linearised
operator L and contaminates the pseudo-inverse.

Quadratic (3-point) Lagrange interpolation preserves
polynomials of degree 2, making all four collision invariants
exact null modes of the discrete operator. We build L with
3-pt interpolation here and verify Ker(L) is clean before
inverting.

Bobylev-Palczewski-Schneider form: operator built in h-form
with on-grid ρ_β^(0)(v_l) prefactor for exact discrete
detailed balance. Combined with 3-pt Lagrange, all four
invariants are null modes to 1e-16.

Source for interdiffusion (∂_s Δn = 1, Δn = (n_L-n_H)/2):
  b_α(v_j) = sgn(α) · v_j / n_α^0  (h-form)
Fickian flux: D_LH = -(1/2)∫ v [ρ_L^(0) h_L - ρ_H^(0) h_H] dv

Final result at L=12.57, σ=0.00314, ρ=31.8, η=0.1, T=1:
  m_max | 1.5  | 2    | 5    | 10   | 20   | 100
  D_LH  | 4.67 | 4.45 | 4.11 | 4.04 | 4.02 | 4.03
D essentially flat — dominated by label-exchange relaxation,
not mass disparity.
=#
push!(LOAD_PATH, @__DIR__)
using GHDBoltzmannSolver
using Printf
using LinearAlgebra

const L_arc = 12.57
const sigma_val = 0.00314
const rho_tot = 400/L_arc
const N_v = 160
const V_max = 8.0

println("="^70)
println("Chapman-Enskog D_LH for binary GHD-Boltzmann")
println("3-point Lagrange interpolation (Mieussens-conservative)")
println("="^70)
@printf("L = %.4f, σ = %.5f, ρ_tot = %.3f, η = %.3f\n",
        L_arc, sigma_val, rho_tot, sigma_val*rho_tot)
@printf("N_v = %d, V_max = %.2f, dv = %.4f\n", N_v, V_max, 2*V_max/N_v)

# -----------------------------------------------------------------
# 3-point Lagrange interpolation kernel
# -----------------------------------------------------------------
@inline function elastic_post(v1, v2, m1, m2)
    M = m1 + m2
    v1p = ((m1 - m2)*v1 + 2*m2*v2)/M
    v2p = (2*m1*v1 + (m2 - m1)*v2)/M
    return v1p, v2p
end

"""
3-pt Lagrange weights at offset δ ∈ [-0.5, 0.5] from central node.
Stencil: {-1, 0, +1} relative spacing 1. Returns (w_-, w_0, w_+).
Preserves polynomials of degree ≤ 2 exactly.
"""
@inline function lagrange3(δ::T) where T
    w_m = δ*(δ - T(1))/T(2)
    w_0 = T(1) - δ*δ
    w_p = δ*(δ + T(1))/T(2)
    return w_m, w_0, w_p
end

"""
Locate v_star on the grid v = -V_max + (j-1/2)*Δv. Returns
central index j_c (1 ≤ j_c-1 and j_c+1 ≤ N_v) and offset δ.
"""
@inline function locate3(v_star::T, V_max::T, dv::T, N_v::Int) where T
    pos = (v_star + V_max)/dv + T(0.5)  # continuous index (1-based, integer at center)
    j_c = round(Int, pos)
    j_c = clamp(j_c, 2, N_v - 1)
    δ = pos - j_c
    return j_c, δ
end

"""
Build the linearised collision operator L̃ : R^{2N_v} → R^{2N_v}
acting on h = (h_L; h_H), the deviation in h-form
(ρ_α = ρ_α^(0) (1 + h_α)).

  L̃[h](α, v_j) = σ Σ_l Δv |v_j-v_l| ρ_β^(0)(v_l) ·
                 [h_α^interp(v_*α) + h_β^interp(v_*β)
                  - h_α(v_j) - h_β(v_l)]

Detailed balance is enforced exactly by using the ON-GRID value
ρ_β^(0)(v_l) as the rate prefactor (Bobylev-Palczewski-Schneider
form). The interpolation acts only on h, so 3-pt Lagrange
preserves polynomials up to degree 2 → all four collision
invariants {1_α, m_α v, m_α v²/2} are exact null modes of L̃.
"""
function build_L_matrix_3pt(N_v::Int, V_max::T, dv::T, sigma::T,
                            m_L::T, m_H::T, v_grid::Vector{T},
                            f_L_eq::Vector{T}, f_H_eq::Vector{T}) where T
    L̃ = zeros(T, 2N_v, 2N_v)
    @inbounds for j in 1:N_v, l in 1:N_v
        diff = abs(v_grid[j] - v_grid[l])
        # Rate coefficient using ON-GRID ρ_β^(0) (exact detailed balance):
        rate_LH = sigma * dv * diff * f_H_eq[l]   # for L-output: β=H, on-grid at v_l
        rate_HL = sigma * dv * diff * f_L_eq[l]   # for H-output: β=L

        # ===== L-species output at v_j: collision (L,v_j) + (H,v_l) =====
        # Skip entire collision pair if pre-velocity falls off-grid, to
        # maintain gain/loss balance (necessary for clean null space).
        v_sL, v_sH = elastic_post(v_grid[j], v_grid[l], m_L, m_H)
        in_range_L = (v_grid[2] ≤ v_sL ≤ v_grid[N_v-1]) &&
                     (v_grid[2] ≤ v_sH ≤ v_grid[N_v-1])
        if in_range_L
            jcL, δL = locate3(v_sL, V_max, dv, N_v)
            jcH, δH = locate3(v_sH, V_max, dv, N_v)
            wLm, wL0, wLp = lagrange3(δL)
            wHm, wH0, wHp = lagrange3(δH)
            L̃[j, jcL-1] += rate_LH * wLm
            L̃[j, jcL  ] += rate_LH * wL0
            L̃[j, jcL+1] += rate_LH * wLp
            L̃[j, N_v + jcH-1] += rate_LH * wHm
            L̃[j, N_v + jcH  ] += rate_LH * wH0
            L̃[j, N_v + jcH+1] += rate_LH * wHp
            # LOSS only if gain is included:
            L̃[j, j]       -= rate_LH
            L̃[j, N_v + l] -= rate_LH
        end

        # ===== H-species output at v_j: collision (H,v_j) + (L,v_l) =====
        v_sHp, v_sLp = elastic_post(v_grid[j], v_grid[l], m_H, m_L)
        in_range_H = (v_grid[2] ≤ v_sHp ≤ v_grid[N_v-1]) &&
                     (v_grid[2] ≤ v_sLp ≤ v_grid[N_v-1])
        if in_range_H
            jcH, δH = locate3(v_sHp, V_max, dv, N_v)
            jcL, δL = locate3(v_sLp, V_max, dv, N_v)
            wHm, wH0, wHp = lagrange3(δH)
            wLm, wL0, wLp = lagrange3(δL)
            L̃[N_v + j, N_v + jcH-1] += rate_HL * wHm
            L̃[N_v + j, N_v + jcH  ] += rate_HL * wH0
            L̃[N_v + j, N_v + jcH+1] += rate_HL * wHp
            L̃[N_v + j, jcL-1] += rate_HL * wLm
            L̃[N_v + j, jcL  ] += rate_HL * wL0
            L̃[N_v + j, jcL+1] += rate_HL * wLp
            L̃[N_v + j, N_v + j] -= rate_HL
            L̃[N_v + j, l]       -= rate_HL
        end
    end
    return L̃
end

function compute_D_LH(m_L::T, m_H::T, kT::T, sigma::T, n_L::T, n_H::T;
                       N_v::Int=N_v, V_max::T=T(V_max)) where T
    dv = 2*V_max/N_v
    v_grid = T[-V_max + (j-T(0.5))*dv for j in 1:N_v]
    Z_L = sqrt(2π*kT/m_L); Z_H = sqrt(2π*kT/m_H)
    f_L = T[(n_L/Z_L)*exp(-m_L*v_grid[j]^2/(2*kT)) for j in 1:N_v]
    f_H = T[(n_H/Z_H)*exp(-m_H*v_grid[j]^2/(2*kT)) for j in 1:N_v]

    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma, m_L, m_H, v_grid, f_L, f_H)

    # Verify null space: L̃·ψ_a should be ≈ 0 in h-form
    # ψ_1 = δ_αL (h_L=1, h_H=0); ψ_2 = δ_αH; ψ_3 = m_α v; ψ_4 = m_α v²/2
    Ψ = zeros(T, 2N_v, 4)
    for j in 1:N_v
        Ψ[j, 1] = T(1)
        Ψ[N_v + j, 2] = T(1)
        Ψ[j, 3] = m_L * v_grid[j]
        Ψ[N_v + j, 3] = m_H * v_grid[j]
        Ψ[j, 4] = T(0.5) * m_L * v_grid[j]^2
        Ψ[N_v + j, 4] = T(0.5) * m_H * v_grid[j]^2
    end
    Lop = opnorm(L̃, 1)
    null_errs = [norm(L̃*Ψ[:, a])/(norm(Ψ[:, a]) * Lop) for a in 1:4]

    # Source in h-form: S_α(v_j) = sgn(α) · v_j / n_α^0
    # Mask the source where f^eq is negligible (physically: h^(1) is
    # undefined where there are no particles to perturb). The CE
    # transport integral is insensitive to these regions because the
    # flux carries a factor ρ^(0).
    fL_max = maximum(f_L); fH_max = maximum(f_H)
    f_thresh = T(1e-8)
    b = zeros(T, 2N_v)
    for j in 1:N_v
        if f_L[j] > f_thresh * fL_max
            b[j]   = +v_grid[j] / n_L
        end
        if f_H[j] > f_thresh * fH_max
            b[N_v + j] = -v_grid[j] / n_H
        end
    end

    # Symmetrise in f^eq inner product. Define D = diag(sqrt(f^eq)).
    # Then S = D L̃ D^{-1} is symmetric, and we solve via its eigendecomp.
    d = T[sqrt(max(f_L[j], T(1e-300))) for j in 1:N_v]
    append!(d, T[sqrt(max(f_H[j], T(1e-300))) for j in 1:N_v])
    D_inv = T(1) ./ d
    # Symmetric form: S = D · L̃ · D^{-1} acts on H = D·h
    S_mat = (d .* L̃) .* D_inv'
    S_sym = (S_mat + S_mat')/T(2)  # enforce symmetry
    eig_vals = eigen(S_sym)
    λ = eig_vals.values
    Q = eig_vals.vectors
    # Show smallest |λ|
    sv_min = sort(abs.(λ))[1:6]

    # Source in symmetric space: B = D · b
    B = d .* b

    # Pseudo-inverse with threshold relative to max |λ|
    λ_max = maximum(abs.(λ))
    rel_thresh = T(1e-10)
    keep = abs.(λ) .> rel_thresh * λ_max
    n_null = count(.!keep)
    λ_inv = [keep[i] ? T(1)/λ[i] : zero(T) for i in 1:length(λ)]
    H_sol = Q * (λ_inv .* (Q' * B))
    h = D_inv .* H_sol

    # Fickian interdiffusion: number flux of species composition.
    # J_Δ = (1/2) ∫ v [ρ_L^(0) h_L - ρ_H^(0) h_H] dv
    #     = -D_LH ∂_s Δn  with our normalisation ∂_s Δn = 1.
    J = zero(T)
    for j in 1:N_v
        J += T(0.5) * v_grid[j] * (f_L[j]*h[j] - f_H[j]*h[N_v + j]) * dv
    end
    D_LH = -J

    return D_LH, null_errs, sv_min, n_null
end

# -----------------------------------------------------------------
# Sweep m_max
# -----------------------------------------------------------------
n_each = rho_tot/2
k1 = 2π/L_arc

println("\nNull-space cleanliness check (should be ≪ 1):")
println("  ||M·ψ_a|| / (||ψ_a|| ||M||) for ψ = {1_L, 1_H, mv, mv²/2}")
println()
println("| m_max | D_LH | γ_sound=D·k² | null_errs (4) | n_null (≥4 ok) |")
println("|---|---|---|---|---|")

results = []
for m_max_val in [2.0, 5.0, 10.0, 20.0, 100.0]
    D, ne, svm, nn = compute_D_LH(1.0, m_max_val, 1.0, sigma_val, n_each, n_each)
    γ_sound = D * k1^2
    @printf("| %g | %.4e | %.4e | [%.1e, %.1e, %.1e, %.1e] | %d |\n",
            m_max_val, D, γ_sound, ne[1], ne[2], ne[3], ne[4], nn)
    push!(results, (m_max_val, D, γ_sound))
end

println("\nMD damping rates (P2/P3 reference): γ_obs ~ 0.02–0.03 at k=k1")
println()
println("Tabulated results for paper:")
println("| m_max | m̄  | D_LH | γ_sound·L (decay length/L) |")
for (m, D, γ) in results
    mbar = (1+m)/2
    println(@sprintf("| %g | %.1f | %.4e | %.4e |", m, mbar, D, γ*L_arc))
end
