#=
Reduced-order hydrodynamic dispersion ω(k) by projection onto
the 4D kernel of the linearised collision operator.

The 4 hydrodynamic invariants {ψ_a}_{a=1..4} = {1_L, 1_H, m_α v, m_α v²/2}
span Ker(L̃) and define the slow manifold of the binary GHD-Boltzmann.
The Euler matrix in this basis (using f^eq inner product) is:

  M_Euler[a,b] = ⟨ψ_a, A ψ_b⟩_{f^eq}

where ⟨·⟩_{f^eq} = ∑_α ∫ dv f^(eq)_α (·) (·) is the natural inner product
in which L̃ is symmetric. Eigenvalues of M_Euler give the sound speeds
(2 modes, ±c_s) and the diffusive limit (heat, composition → 0).

At O(k²) the dissipative correction comes from
  M_diss[a,b] = -i · k² · ⟨A ψ_a, L̃^{-1}_⊥ A ψ_b⟩_{f^eq}

with L̃^{-1}_⊥ restricted to the orthogonal complement of Ker(L̃).
This gives the Navier-Stokes hydrodynamic matrix with sound damping
extracted as -Im(ω_sound)/k².
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")
include("dispersion_relation.jl")  # for build_streaming_A
using LinearAlgebra, Printf

println()
println("="^70)
println("Reduced hydrodynamic dispersion via 4D projection")
println("="^70)

function hydro_dispersion(m_max::Float64; n_each=rho_tot/2, kT=1.0,
                           N_v=N_v, V_max=V_max, sigma=sigma_val)
    dv = 2*V_max/N_v
    v = [-V_max + (j-0.5)*dv for j in 1:N_v]
    Z_L = sqrt(2π*kT); Z_H = sqrt(2π*kT/m_max)
    f_L = [(n_each/Z_L)*exp(-v[j]^2/(2*kT)) for j in 1:N_v]
    f_H = [(n_each/Z_H)*exp(-m_max*v[j]^2/(2*kT)) for j in 1:N_v]
    η_0 = sigma * 2 * n_each

    # L̃ and A
    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma, 1.0, m_max, v, f_L, f_H)
    A, _, _, _, _, _ = build_streaming_A(m_max; n_each=n_each, kT=kT,
                                          N_v=N_v, V_max=V_max, sigma=sigma)

    # 4 hydrodynamic invariants ψ_a in h-form (columns of Ψ matrix)
    Ψ = zeros(2*N_v, 4)
    for j in 1:N_v
        Ψ[j, 1] = 1.0
        Ψ[N_v+j, 2] = 1.0
        Ψ[j, 3] = v[j]
        Ψ[N_v+j, 3] = m_max * v[j]
        Ψ[j, 4] = 0.5*v[j]^2
        Ψ[N_v+j, 4] = 0.5*m_max*v[j]^2
    end

    # Inner-product weight matrix W = diag(f^eq dv) on h-form
    w = vcat(f_L .* dv, f_H .* dv)
    # ⟨a, b⟩_{f^eq} = Σ a_i w_i b_i

    # Gram matrix of invariants
    Gram = zeros(4, 4)
    for a in 1:4, b in 1:4
        Gram[a, b] = sum(w[i] * Ψ[i, a] * Ψ[i, b] for i in 1:2*N_v)
    end

    # Euler matrix M_E[a, b] = ⟨ψ_a, A ψ_b⟩_{f^eq} solved in dual basis
    AΨ = A * Ψ  # 2N_v × 4
    M_E_unscaled = zeros(4, 4)
    for a in 1:4, b in 1:4
        M_E_unscaled[a, b] = sum(w[i] * Ψ[i, a] * AΨ[i, b] for i in 1:2*N_v)
    end
    M_E = Gram \ M_E_unscaled  # solve Gram·M_E = M_E_unscaled

    # Dissipative correction: M_diss[a,b] = -⟨A ψ_a, L̃^{-1}_⊥ A ψ_b⟩
    # Solve L̃ x_b = A ψ_b on the orthogonal complement of Ker(L̃)
    # Use symmetric eigendecomp in f^eq metric
    d = sqrt.(max.(w, 1e-300))
    D_inv = 1.0 ./ d
    S_mat = (d .* L̃) .* D_inv'
    S_sym = (S_mat + S_mat')/2
    ev = eigen(S_sym)
    λ_eig = ev.values; Q = ev.vectors
    λ_max_abs = maximum(abs.(λ_eig))
    keep = abs.(λ_eig) .> 1e-6 * λ_max_abs
    λ_inv = [keep[i] ? 1.0/λ_eig[i] : 0.0 for i in 1:length(λ_eig)]

    # x_b = L̃^{-1}_⊥ (A ψ_b)  -- in h-form
    function solve_perp(b_vec)
        B = d .* b_vec
        H_sol = Q * (λ_inv .* (Q' * B))
        return D_inv .* H_sol
    end

    M_D_unscaled = zeros(4, 4)
    Xb = zeros(2*N_v, 4)
    for b in 1:4
        Xb[:, b] = solve_perp(AΨ[:, b])
    end
    AXb = A * Xb
    for a in 1:4, b in 1:4
        # ⟨A ψ_a, x_b⟩_{f^eq} = inner product of A ψ_a with x_b
        # = ⟨ψ_a, A^T x_b⟩... but in f^eq metric A is not symmetric
        # We want ⟨ψ_a, A x_b⟩ for the standard dispersion expansion
        M_D_unscaled[a, b] = sum(w[i] * Ψ[i, a] * AXb[i, b] for i in 1:2*N_v)
    end
    M_D = Gram \ M_D_unscaled

    return M_E, M_D, Gram
end

# Eigenmodes of the hydrodynamic matrix at finite k
function find_sound_mode(M_E, M_D, k::Float64)
    M_total = k * M_E + im * k^2 * M_D
    eig = eigen(M_total)
    return eig.values, eig.vectors
end

# Report
m_max_vals = (2.0, 5.0, 10.0, 100.0)
k1 = 2π/L_arc

println()
println("4×4 Euler matrix M_E (Re part), at u_0=0 background:")
for m_max in m_max_vals
    M_E, M_D, Gram = hydro_dispersion(m_max)
    m̄ = (1+m_max)/2
    c_s_an = sqrt(3/m̄)/0.9
    println("\n--- m_max = $m_max  (c_s_an = $(round(c_s_an,digits=4))) ---")
    println("Eigenvalues of M_E (sound speeds c_s):")
    eig_E = eigvals(M_E)
    for ω_E in sort(real.(eig_E), rev=true)
        @printf("  c_s_E = %+.5f  (ratio to c_s_an = %.4f)\n",
                ω_E, ω_E/c_s_an)
    end

    println("k-sweep dispersion (track sound mode at positive Re):")
    println("    k       | Re(ω)    | Im(ω)    | c_s=Re/k | γ=-Im/k²")
    for k in (0.01, 0.05, 0.10, 0.25, k1, 1.0)
        ωs, _ = find_sound_mode(M_E, M_D, k)
        ω_sound = ωs[argmax(real.(ωs))]
        c_s_eff = real(ω_sound)/k
        γ_eff = -imag(ω_sound)/k^2
        @printf("    %.3f   | %+.5f | %+.4e | %.5f | %.4e\n",
                k, real(ω_sound), imag(ω_sound), c_s_eff, γ_eff)
    end
end
