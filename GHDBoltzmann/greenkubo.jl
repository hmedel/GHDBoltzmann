#=
Green-Kubo formula for the wavenumber-dependent thermal
conductivity λ_T(k, m_max) of the linearised binary
GHD-Boltzmann operator. The key step is projecting onto
the heat-current OPERATOR, not selecting modes by their
imaginary part (which mis-identifies the heat mode as the
composition mode at finite k, per hyperscaling_sweep.jl).

Green-Kubo:
  λ_T(k) = (1/T²) Im⟨J_E, M(k)^{-1} J_E⟩_{ρ^(0)}

where M(k) = k·A + i·L̃, J_E^(α)(v) = v(m_α v²/2 - 3T/2)
is the heat-current operator (energy flux minus enthalpy
times mass flux in 1D ideal gas), and ⟨·,·⟩_{ρ^(0)} is
the f^eq-weighted inner product.

Sanity check at k=0:
  M(0) = i·L̃
  λ_T(0) = -Re⟨J_E, L̃^{-1}_⊥ J_E⟩/T²
  = standard Chapman-Enskog λ_T from Sec V

Hyperscaling test:
  Plot κ(k, m) = λ_T(k, m)/(ρ c_p)
  vs (m-1)·k^{-ζ} with rescaling λ_T·k^β
  If collapse holds, hyperscaling verified.
=#
push!(LOAD_PATH, @__DIR__)
include("chapman_enskog.jl")
include("dispersion_relation.jl")
using LinearAlgebra, Printf

const T_kT = 1.0
const ρ_cp = 3.0/2.0 * rho_tot * T_kT  # 1D ideal gas heat capacity per length

function greenkubo_lambdaT(m_max::Float64, k::Float64;
                            n_each=rho_tot/2, kT=T_kT,
                            N_v::Int=N_v, V_max::Float64=V_max,
                            sigma=sigma_val)
    dv = 2*V_max/N_v
    v = [-V_max + (j-0.5)*dv for j in 1:N_v]
    Z_L = sqrt(2π*kT); Z_H = sqrt(2π*kT/m_max)
    fL = [(n_each/Z_L)*exp(-v[j]^2/(2*kT)) for j in 1:N_v]
    fH = [(n_each/Z_H)*exp(-m_max*v[j]^2/(2*kT)) for j in 1:N_v]

    L̃ = build_L_matrix_3pt(N_v, V_max, dv, sigma, 1.0, m_max, v, fL, fH)
    A, _, _, _, _, _ = build_streaming_A(m_max; n_each=n_each, kT=kT,
                                          N_v=N_v, V_max=V_max, sigma=sigma)
    M = k*A + im*L̃

    # Heat-current operator J_E^(α)(v) = v(m_α v²/2 - 3T/2)
    J_E = zeros(2*N_v)
    for j in 1:N_v
        J_E[j]       = v[j] * (1.0  *v[j]^2/2 - 1.5*kT)
        J_E[N_v+j]   = v[j] * (m_max*v[j]^2/2 - 1.5*kT)
    end

    # Inner-product weight: W = diag(ρ^(0) dv)
    w = vcat(fL .* dv, fH .* dv)
    Wf = w .* J_E  # right-hand side: W · J_E

    # Solve M y = Wf for y
    if k ≈ 0.0
        # At k=0, M = iL̃, singular (4 zero modes). Use pseudoinverse.
        d = sqrt.(max.(w, 1e-300))
        D_inv = 1.0 ./ d
        S_mat = (d .* L̃) .* D_inv'
        S_sym = (S_mat + S_mat')/2
        ev = eigen(S_sym)
        λ_eig = ev.values; Q = ev.vectors
        # Symmetrised RHS: B = D · J_E (without W)
        B = d .* J_E
        λ_max_abs = maximum(abs.(λ_eig))
        keep = abs.(λ_eig) .> 1e-6 * λ_max_abs
        # M(0) = iL̃, so M^{-1} = -i L̃^{-1}_⊥
        # Im⟨J_E, M^{-1} J_E⟩_W = Im(-i Re⟨J_E, L̃^{-1}_⊥ J_E⟩_W) = -Re⟨J_E, L̃^{-1}_⊥ J_E⟩_W
        L_inv_J = D_inv .* (Q * ([keep[i] ? 1.0/λ_eig[i] : 0.0 for i in 1:length(λ_eig)] .* (Q' * B)))
        return -sum(J_E .* w .* L_inv_J) / kT^2
    else
        # Full linear solve
        y = M \ Wf
        gk = J_E' * y  # ⟨J_E, M^{-1} W J_E⟩... wait actually this isn't right
        # Want ⟨J_E, M^{-1} J_E⟩_W = J_E' W (M^{-1} J_E)
        # = (W J_E)' (M^{-1} J_E)
        # Solve M x = J_E (without W), then result = J_E' W x
        x = M \ J_E
        gk = sum(J_E .* w .* x)  # this is ⟨J_E, x⟩_W = ⟨J_E, M^{-1} J_E⟩_W
        return imag(gk) / kT^2
    end
end

# Sanity check at k=0 against C-E λ_T
println("="^70)
println("Sanity check: Green-Kubo at k→0 vs Chapman-Enskog λ_T")
println("="^70)
println("m_max | λ_T^CE (Sec V) | λ_T^GK(k=0)")
for m in (2.0, 5.0, 10.0)
    λ_GK_0 = greenkubo_lambdaT(m, 0.0)
    @printf("%g    | (paper)        | %.2f\n", m, λ_GK_0)
end

println()
println("="^70)
println("Hyperscaling sweep with Green-Kubo (project on J_E)")
println("="^70)

const k_grid = [0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
const m_grid = [1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0]

λT_data = zeros(length(m_grid), length(k_grid))
println("\nλ_T(k, m_max):")
println("m\\k\t" * join([@sprintf("%.3f", k) for k in k_grid], "\t"))
for (i, m) in enumerate(m_grid)
    print("$m\t")
    for (j, k) in enumerate(k_grid)
        λ = greenkubo_lambdaT(m, k)
        λT_data[i, j] = λ
        print(@sprintf("%.2f\t", λ))
    end
    println()
end

# Hyperscaling test: λ_T·k^β  vs  (m-1)·k^{-ζ}
const α_local = 1.53
const β_local = 1/3
const ζ_local = β_local / α_local  # ≈ 0.218

println("\nHyperscaling rescaling: λ_T·k^β vs (m-1)k^{-ζ}")
println("β=$(round(β_local,digits=3)), ζ=$(round(ζ_local,digits=3))")
xs = Float64[]; ys = Float64[]
for (i, m) in enumerate(m_grid), (j, k) in enumerate(k_grid)
    λ = λT_data[i, j]
    if λ > 0
        push!(xs, (m-1.0)*k^(-ζ_local))
        push!(ys, λ*k^β_local)
    end
end

# Bin and check spread
println("\nBy-bin variance of log10(λ·k^β) at fixed (m-1)k^{-ζ}:")
log_x = log10.(xs); log_y = log10.(ys)
n_bins = 6
bins = range(minimum(log_x), maximum(log_x), length=n_bins+1)
spreads = Float64[]
for b in 1:n_bins
    mask = (log_x .≥ bins[b]) .& (log_x .< bins[b+1])
    if count(mask) >= 2
        σ = sqrt(sum((log_y[mask] .- sum(log_y[mask])/count(mask)).^2)/(count(mask)-1))
        push!(spreads, σ)
        @printf("  bin [%.2f, %.2f]: n=%d, std(log10 y) = %.3f\n",
                bins[b], bins[b+1], count(mask), σ)
    end
end
@printf("\nMean within-bin std: %.3f dex (collapse ≤ 0.1 dex)\n",
        sum(spreads)/length(spreads))

# Plot
using Plots, LaTeXStrings
gr()
default(framestyle=:box, gridstyle=:dot, gridlinewidth=0.4,
        size=(450, 330), titlefontsize=11, labelfontsize=10,
        legendfontsize=8, tickfontsize=8)
colors = palette(:viridis, length(m_grid))
p = plot(xscale=:log10, yscale=:log10,
         xlabel=L"(m_{\max}-1)\,k^{-\zeta}",
         ylabel=L"\lambda_T\,k^{\beta}",
         title=L"\rm Green-Kubo\ collapse\ test:\ \beta=1/3,\ \zeta=0.218",
         legend=:topleft)
for (i, m) in enumerate(m_grid)
    xs_m = Float64[]; ys_m = Float64[]
    for (j, k) in enumerate(k_grid)
        λ = λT_data[i, j]
        if λ > 0
            push!(xs_m, (m-1.0)*k^(-ζ_local))
            push!(ys_m, λ*k^β_local)
        end
    end
    scatter!(xs_m, ys_m, ms=5, color=colors[i],
             label=@sprintf("m=%.1f", m))
end
savefig(p, joinpath(@__DIR__, "..", "Paper02", "figures",
                     "fig_hyperscaling_gk.pdf"))
println("\nFigure: fig_hyperscaling_gk.pdf")
