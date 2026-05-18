#=
Diagnose mass-flux structure of the CE solution for the binary
species-composition perturbation. Check whether
  m_L j_L + m_H j_H = 0  (total mass flux from diffusion alone vanishes)
should hold by overall mass + momentum conservation.

Print j_L, j_H, and their mass-weighted sum.
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")
using LinearAlgebra, Printf

println()
println("DIAGNOSTIC: kinetic structure of CE D_LH for binary species mode")
println("="^70)

for m_max in (2.0, 5.0, 10.0, 100.0)
    p = GHDBParams(L=L_arc, sigma=sigma_val, m_L=1.0, m_H=m_max, kT=1.0,
                   N_s=10, N_v=N_v, V_max=V_max, cfl=0.25)
    st = GHDBState(p)
    nL = nH = rho_tot/2
    Z_L = sqrt(2π); Z_H = sqrt(2π/m_max)
    fL = [(nL/Z_L)*exp(-st.v[j]^2/2) for j in 1:N_v]
    fH = [(nH/Z_H)*exp(-m_max*st.v[j]^2/2) for j in 1:N_v]
    L̃ = build_L_matrix_3pt(N_v, V_max, 2*V_max/N_v, sigma_val, 1.0, m_max,
                           st.v, fL, fH)

    b = zeros(2N_v)
    for j in 1:N_v
        b[j]     = +st.v[j]/nL
        b[N_v+j] = -st.v[j]/nH
    end

    d = vcat([sqrt(max(fL[j], 1e-300)) for j in 1:N_v],
             [sqrt(max(fH[j], 1e-300)) for j in 1:N_v])
    D_inv = 1.0 ./ d
    S_mat = (d .* L̃) .* D_inv'
    S_sym = (S_mat + S_mat')/2
    ev = eigen(S_sym)
    λ = ev.values; Q = ev.vectors
    B = d .* b
    λ_max = maximum(abs.(λ))
    # Threshold sweep to find the right stopping point
    for thresh in (1e-3, 1e-5, 1e-7, 1e-10, 1e-12)
        keep = abs.(λ) .> thresh*λ_max
        λ_inv = [keep[i] ? 1.0/λ[i] : 0.0 for i in 1:length(λ)]
        h = D_inv .* (Q * (λ_inv .* (Q' * B)))
        dv = 2*V_max/N_v
        j_L = sum(st.v[j] * fL[j] * h[j] for j in 1:N_v) * dv
        j_H = sum(st.v[j] * fH[j] * h[N_v+j] for j in 1:N_v) * dv
        mom = 1.0 * j_L + m_max * j_H
        D_LH_my = -0.5 * (j_L - j_H)
        n_keep = count(keep)
        @printf("  thresh=%.0e  n_keep=%d  j_L=%+.2e  j_H=%+.2e  m_L·j_L+m_H·j_H=%+.2e  D_LH=%.4f\n",
                thresh, n_keep, j_L, j_H, mom, D_LH_my)
    end
    # Best (canonical) threshold
    keep = abs.(λ) .> 1e-10*λ_max
    λ_inv = [keep[i] ? 1.0/λ[i] : 0.0 for i in 1:length(λ)]
    h = D_inv .* (Q * (λ_inv .* (Q' * B)))

    dv = 2*V_max/N_v
    j_L = sum(st.v[j] * fL[j] * h[j] for j in 1:N_v) * dv
    j_H = sum(st.v[j] * fH[j] * h[N_v+j] for j in 1:N_v) * dv
    mom = 1.0 * j_L + m_max * j_H
    D_LH_my = -0.5 * (j_L - j_H)

    @printf("m_max=%g\n", m_max)
    @printf("  j_L = %+.5f    j_H = %+.5f\n", j_L, j_H)
    @printf("  m_L·j_L + m_H·j_H = %+.4e  (should ≈ 0 if mom conserved)\n", mom)
    @printf("  D_LH(my) = -(j_L - j_H)/2 = %.4f\n", D_LH_my)
    @printf("  D_self_L = -j_L = %.4f\n", -j_L)
    @printf("  Ratio (m_H/m_L) j_H / j_L = %+.4f  (should = -m_L/m_H = %.4f)\n",
            (m_max/1.0)*j_H/j_L, -1.0/m_max)
    println()
end
