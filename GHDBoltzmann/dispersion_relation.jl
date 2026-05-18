#=
Full dispersion relation ω(k) for the linearised GHD-Boltzmann
equation. The eigenvalue problem in Fourier (mode k) is:

  ω h̃ = (k·A + i·L̃) h̃

where L̃ is the linearised collision operator (h-form, 3-pt Lagrange
+ Bobylev) computed in Sec V, and A is the streaming operator:

  A[h]_α(v) = (v/(1-η₀))·h_α(v)
            + (vσ/(1-η₀)²)·M_n[h]
            - (η₀/(1-η₀))·M_u[h]

with the rank-2 moment corrections:
  M_n[h] = ∫ Σ_β ρ_β^(0)(v) h_β(v) dv   (total number perturbation)
  M_u[h] = (1/ρ_tot) ∫ v Σ_β ρ_β^(0)(v) h_β(v) dv   (number-weighted ū)

The dressed-velocity rank-2 corrections enter at uniform background
because v_eff depends on (ρ_tot, ū) which are NON-LOCAL moments.

For small k, identify the sound mode as the eigenvector with the
largest |Re(ω)|; extract c_s = Re(ω)/k and γ_sound = -Im(ω)/k².
Compare with empirical γ_DVM = 0.042 from Sec VI.
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")
using LinearAlgebra, Printf

println("="^70)
println("Full dispersion relation ω(k) for linearised GHD-Boltzmann")
println("="^70)

function build_streaming_A(m_max::Float64; n_each=rho_tot/2, kT=1.0,
                            N_v=N_v, V_max=V_max, sigma=sigma_val)
    dv = 2*V_max/N_v
    v = [-V_max + (j-0.5)*dv for j in 1:N_v]
    Z_L = sqrt(2π*kT); Z_H = sqrt(2π*kT/m_max)
    f_L = [(n_each/Z_L)*exp(-v[j]^2/(2*kT)) for j in 1:N_v]
    f_H = [(n_each/Z_H)*exp(-m_max*v[j]^2/(2*kT)) for j in 1:N_v]
    η_0 = sigma * 2 * n_each

    # Build the streaming operator A
    # A = diag(v/(1-η₀)) on h
    #   + outer products: (vσ/(1-η₀)²) * ⟨1, ρ^(0) ·⟩  - (η₀/(1-η₀)) * ⟨v/ρ_tot, ρ^(0) ·⟩
    A = zeros(2*N_v, 2*N_v)
    for j in 1:N_v
        A[j, j]         = v[j] / (1 - η_0)
        A[N_v+j, N_v+j] = v[j] / (1 - η_0)
    end
    # Moment functionals (as row vectors weighted by ρ^(0) and dv):
    Mn = vcat(f_L .* dv, f_H .* dv)  # ⟨h, ρ^(0)⟩ = δn_tot
    Mu = vcat(v .* f_L .* dv, v .* f_H .* dv) ./ (2*n_each)  # ⟨h, vρ^(0)/ρ_tot⟩
    # Add rank-2 corrections:
    pref_n = sigma/(1 - η_0)^2
    pref_u = η_0/(1 - η_0)
    for r in 1:2*N_v
        v_row = r <= N_v ? v[r] : v[r-N_v]
        for c in 1:2*N_v
            A[r, c] += v_row * pref_n * Mn[c] - pref_u * Mu[c]
        end
    end
    return A, f_L, f_H, v, dv, η_0
end

function dispersion(m_max::Float64, k::Float64; verbose=false)
    A, f_L, f_H, v, dv, η_0 = build_streaming_A(m_max)
    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma_val, 1.0, m_max, v, f_L, f_H)

    M = k * A + im * L̃
    ω = eigvals(M)

    # Identify the SOUND mode (hydrodynamic, collective): the eigenvalue
    # closest to the expected hydrodynamic prediction ±c_s_an·k.
    # Free-streaming/kinetic modes have |ω| ~ k·V_max, much larger.
    m̄ = (1.0 + m_max)/2
    c_s_an = sqrt(3/m̄)/(1 - η_0)
    target_pos = c_s_an * k
    target_neg = -c_s_an * k

    # Distance to target, looking for least-damped mode within hydro subspace
    dist_pos = abs.(real.(ω) .- target_pos) .+ 10*abs.(imag.(ω))  # penalise damping less
    dist_neg = abs.(real.(ω) .- target_neg) .+ 10*abs.(imag.(ω))

    i_pos = argmin(dist_pos)
    i_neg = argmin(dist_neg)
    ω_sound_pos = ω[i_pos]
    ω_sound_neg = ω[i_neg]
    return ω_sound_pos, ω_sound_neg, ω
end

# Sound speed prediction at small k
k1_disp = 2π/L_arc
println()
println("Continuum sound speed at η=0.1, m_max=5: c_s_an = ", sqrt(3/3.0)/0.9)
println()

for m_max in (2.0, 5.0, 10.0, 100.0)
    m̄ = (1.0 + m_max)/2
    c_s_an = sqrt(3/m̄) / (1 - 0.1)

    println("\nm_max = $m_max  (c_s_an = $(round(c_s_an, digits=4)))")
    println("k         | Re(ω_sound) | Im(ω_sound)  | c_s = Re(ω)/k | γ = -Im(ω)/k²")
    println("─"^80)
    for k in (0.05, 0.10, 0.25, 0.50, k1_disp, 1.0)
        ω_pos, ω_neg, _ = dispersion(m_max, k)
        c_s_obs = real(ω_pos) / k
        γ = -imag(ω_pos) / k^2
        @printf("%.3f     | %+.5f    | %+.5e | %.4f       | %.4e\n",
                k, real(ω_pos), imag(ω_pos), c_s_obs, γ)
    end
end

# Detailed analysis at our experimental k_1
println()
println("="^70)
println("All 4 hydrodynamic modes at k = k₁ = 2π/L = $(round(k1_disp,digits=4))")
println("Lowest 8 |Im(ω)| eigenvalues (slow modes)")
println("="^70)

for m_max in (5.0,)
    _, _, ω = dispersion(m_max, k1_disp)
    perm = sortperm(abs.(imag.(ω)))
    println("\nm_max=$m_max:")
    println("  ω = Re + iIm  → c_s_eff = Re/k₁, γ_eff = -Im/k₁²")
    for i in 1:8
        ω_i = ω[perm[i]]
        @printf("    %+10.5f %+10.5f i   (c_s_eff=%+.4f, γ_eff=%+.4e)\n",
                real(ω_i), imag(ω_i), real(ω_i)/k1_disp, -imag(ω_i)/k1_disp^2)
    end
end
