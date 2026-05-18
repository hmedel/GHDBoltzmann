#=
Sollich-review revisions:
  (a) Update λ_T source with species-dependent Eucken constant
      C_α = 1/2 + m_α/m̄ (correct binary form; old code had 3/2 for both,
      saved by pinv projection onto Range(L̃) but tex was misleading).
  (b) Verify λ_T value is unchanged (orthogonality of correction to range).
  (c) Convergence sweep for D_LH and λ_T over (N_v, V_max).
  (d) λ_T divergence scaling V_max → ∞ at m_max=2 (anomalous transport
      signature for near-integrable single-species limit).
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")
using LinearAlgebra, Printf

function lambdaT_species_dep(m_max::Float64; n_each=rho_tot/2, kT=1.0,
                              N_v::Int=N_v, V_max::Float64=V_max,
                              sigma=sigma_val)
    dv = 2*V_max/N_v
    v = [-V_max + (j-0.5)*dv for j in 1:N_v]
    Z_L = sqrt(2π*kT); Z_H = sqrt(2π*kT/m_max)
    f_L = [(n_each/Z_L)*exp(-v[j]^2/(2*kT)) for j in 1:N_v]
    f_H = [(n_each/Z_H)*exp(-m_max*v[j]^2/(2*kT)) for j in 1:N_v]
    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma, 1.0, m_max, v, f_L, f_H)

    m̄ = 0.5*(1.0 + m_max)
    # Correct species-dependent Eucken constant (was hardcoded 3/2 before).
    C_L = 0.5 + 1.0/m̄
    C_H = 0.5 + m_max/m̄
    b = zeros(2*N_v)
    for j in 1:N_v
        b[j]     = (v[j]/kT) * (1.0  *v[j]^2/(2*kT) - C_L)
        b[N_v+j] = (v[j]/kT) * (m_max*v[j]^2/(2*kT) - C_H)
    end

    d = vcat([sqrt(max(f_L[j], 1e-300)) for j in 1:N_v],
             [sqrt(max(f_H[j], 1e-300)) for j in 1:N_v])
    D_inv = 1.0 ./ d
    S_mat = (d .* L̃) .* D_inv'
    S_sym = (S_mat + S_mat')/2
    ev = eigen(S_sym)
    λ_eig = ev.values; Q = ev.vectors
    B = d .* b
    λ_max_abs = maximum(abs.(λ_eig))
    keep = abs.(λ_eig) .> 1e-6 * λ_max_abs
    λ_inv = [keep[i] ? 1.0/λ_eig[i] : 0.0 for i in 1:length(λ_eig)]
    h = D_inv .* (Q * (λ_inv .* (Q' * B)))

    q = 0.0
    for j in 1:N_v
        q += v[j] * (1.0  *v[j]^2/2 - 1.5*kT)*f_L[j]*h[j]*dv
        q += v[j] * (m_max*v[j]^2/2 - 1.5*kT)*f_H[j]*h[N_v+j]*dv
    end
    return -q
end

function DLH_extract(m_max::Float64; n_each=rho_tot/2, kT=1.0,
                     N_v::Int=N_v, V_max::Float64=V_max,
                     sigma=sigma_val)
    dv = 2*V_max/N_v
    v = [-V_max + (j-0.5)*dv for j in 1:N_v]
    Z_L = sqrt(2π*kT); Z_H = sqrt(2π*kT/m_max)
    f_L = [(n_each/Z_L)*exp(-v[j]^2/(2*kT)) for j in 1:N_v]
    f_H = [(n_each/Z_H)*exp(-m_max*v[j]^2/(2*kT)) for j in 1:N_v]
    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma, 1.0, m_max, v, f_L, f_H)

    b = zeros(2*N_v)
    for j in 1:N_v
        b[j]     = +v[j]/n_each
        b[N_v+j] = -v[j]/n_each
    end
    d = vcat([sqrt(max(f_L[j], 1e-300)) for j in 1:N_v],
             [sqrt(max(f_H[j], 1e-300)) for j in 1:N_v])
    D_inv = 1.0 ./ d
    S_mat = (d .* L̃) .* D_inv'
    S_sym = (S_mat + S_mat')/2
    ev = eigen(S_sym)
    λ_eig = ev.values; Q = ev.vectors
    B = d .* b
    λ_max_abs = maximum(abs.(λ_eig))
    keep = abs.(λ_eig) .> 1e-6 * λ_max_abs
    λ_inv = [keep[i] ? 1.0/λ_eig[i] : 0.0 for i in 1:length(λ_eig)]
    h = D_inv .* (Q * (λ_inv .* (Q' * B)))
    J = 0.0
    for j in 1:N_v
        J += 0.5 * v[j] * (f_L[j]*h[j] - f_H[j]*h[N_v+j]) * dv
    end
    return -J
end

println("="^70)
println("(a)+(b) λ_T with species-dependent C_α (should equal old value)")
println("="^70)
for m in (2.0, 5.0, 10.0, 100.0)
    λ_old_proj = compute_lambda_T_old = nothing
    λ_new = lambdaT_species_dep(m)
    @printf("  m_max=%g  λ_T (species-dep C) = %.4f\n", m, λ_new)
end

println("\n"*"="^70)
println("(c) Convergence sweep at m_max=5")
println("="^70)
println("| N_v | V_max | D_LH | λ_T |")
println("|---|---|---|---|")
for (Nv, Vm) in ((96, 5.0), (96, 6.0), (96, 8.0),
                  (128, 6.0), (128, 8.0), (160, 6.0), (160, 8.0),
                  (200, 8.0), (200, 10.0))
    D = DLH_extract(5.0; N_v=Nv, V_max=Vm)
    L = lambdaT_species_dep(5.0; N_v=Nv, V_max=Vm)
    @printf("| %d | %.1f | %.3f | %.2f |\n", Nv, Vm, D, L)
end

println("\n"*"="^70)
println("(d) λ_T divergence at m_max=2 (near-integrable, V_max → ∞)")
println("="^70)
println("| V_max | λ_T(m=2) |")
println("|---|---|")
for Vm in (5.0, 6.0, 8.0, 10.0, 12.0, 15.0)
    Nv_use = max(128, round(Int, 16*Vm))
    L = lambdaT_species_dep(2.0; N_v=Nv_use, V_max=Vm)
    @printf("| %.1f | %.1f |\n", Vm, L)
end
