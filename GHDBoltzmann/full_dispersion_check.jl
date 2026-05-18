#=
Verify: full 2N_v eigenvalue problem of M(k) = k·A + i·L̃
at k=k_1, vs the 4D reduced (Euler+NS) projection.

At Kn ~ 1 the reduced projection breaks down. The full
2N_v eigenproblem should give the true sound mode.
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")
include("dispersion_relation.jl")  # has build_streaming_A
using LinearAlgebra, Printf

k1 = 2π/L_arc
println()
println("="^70)
println("Full 2N_v dispersion at k = k_1 = $(round(k1,digits=4))")
println("Comparing to reduced 4D Euler+NS projection")
println("="^70)

for m_max in (5.0,)
    m̄ = 0.5*(1.0 + m_max)
    c_s_an = sqrt(3/m̄)/0.9
    println("\nm_max = $m_max, c_s_an = $(round(c_s_an,digits=4))")
    println("("*"="^60*")")

    # Full 2N_v eigenproblem
    A, fL, fH, v, dv, η_0 = build_streaming_A(m_max)
    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma_val, 1.0, m_max, v, fL, fH)
    M = k1*A + im*L̃
    ω_all = eigvals(M)
    # Filter: hydrodynamic sound mode = positive Re, smallest |Im(ω)|/|Re(ω)|
    # Sweep target Re(ω) values to find what's there
    target_cs = c_s_an
    println("Modes near Re(ω) ≈ c_s_an·k_1 = $(round(target_cs*k1,digits=4)):")
    pos = filter(ω -> real(ω) > 0 && real(ω) < 2*target_cs*k1, ω_all)
    sorted = sort(pos; by=ω -> real(ω))
    for ω in sorted[1:min(8, length(sorted))]
        @printf("  ω = %+.4f %+.5f i  (Re/k_1 = %+.4f, |Im|/k_1² = %+.4f)\n",
                real(ω), imag(ω), real(ω)/k1, -imag(ω)/k1^2)
    end

    println("\nAll Re(ω) > 0 modes sorted by |Im|:")
    pos2 = filter(ω -> real(ω) > 0, ω_all)
    sorted2 = sort(pos2; by=ω -> abs(imag(ω)))
    for ω in sorted2[1:10]
        @printf("  ω = %+.4f %+.5f i  (Re/k_1 = %+.4f)\n",
                real(ω), imag(ω), real(ω)/k1)
    end
end
