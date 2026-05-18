#=
Threshold sweep for λ_T (mirror of audit_DLH.jl).
=#
push!(LOAD_PATH, @__DIR__)
include("thermal_conductivity.jl")
using LinearAlgebra, Printf

println()
println("="^70)
println("λ_T threshold sweep")
println("="^70)

for m_max in (2.0, 5.0, 10.0, 100.0)
    println("\nm_max=$m_max")
    println("  thresh | n_keep | λ_T | q_decomp_L | q_decomp_H")
    dv = 2*V_max/N_v
    v_grid = [-V_max + (j-0.5)*dv for j in 1:N_v]
    Z_L = sqrt(2π); Z_H = sqrt(2π/m_max)
    f_L = [(rho_tot/2/Z_L)*exp(-v_grid[j]^2/2) for j in 1:N_v]
    f_H = [(rho_tot/2/Z_H)*exp(-m_max*v_grid[j]^2/2) for j in 1:N_v]
    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma_val, 1.0, m_max,
                           v_grid, f_L, f_H)
    b = zeros(2*N_v)
    for j in 1:N_v
        v = v_grid[j]
        b[j]       = v * (v^2/2 - 1.5)
        b[N_v+j]   = v * (m_max*v^2/2 - 1.5)
    end
    d = vcat([sqrt(max(f_L[j], 1e-300)) for j in 1:N_v],
             [sqrt(max(f_H[j], 1e-300)) for j in 1:N_v])
    D_inv = 1.0 ./ d
    S_mat = (d .* L̃) .* D_inv'
    S_sym = (S_mat + S_mat')/2
    ev = eigen(S_sym)
    λ = ev.values; Q = ev.vectors
    B = d .* b
    λ_max = maximum(abs.(λ))
    for thresh in (1e-3, 1e-5, 1e-7, 1e-10, 1e-12)
        keep = abs.(λ) .> thresh*λ_max
        λ_inv = [keep[i] ? 1.0/λ[i] : 0.0 for i in 1:length(λ)]
        h = D_inv .* (Q * (λ_inv .* (Q' * B)))
        q_L = 0.0; q_H = 0.0
        for j in 1:N_v
            v = v_grid[j]
            q_L += v * (v^2/2 - 1.5)*f_L[j]*h[j]*dv
            q_H += v * (m_max*v^2/2 - 1.5)*f_H[j]*h[N_v+j]*dv
        end
        @printf("  %.0e | %d | %.4f | %.3f | %.3f\n",
                thresh, count(keep), -(q_L+q_H), q_L, q_H)
    end
end
