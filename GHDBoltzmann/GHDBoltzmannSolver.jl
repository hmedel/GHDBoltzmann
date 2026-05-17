#=
GHDBoltzmannSolver.jl

Discrete-velocity solver for the two-species GHD-Boltzmann
equation of Paper IV [Medel, in preparation], implementing
Sec. III of Paper V. Hard-rod gas with binary mass disorder
on a closed Riemannian curve, reduced to flat arc-length
coordinate.

  ∂_t ρ_α + ∂_s [v_eff_α ρ_α] = C_α[ρ_L, ρ_H]   (α ∈ {L, H})

  v_eff_α = (v - σ ū) / (1 - σ ρ_tot)
  C_α(v) = σ ∫ dv' |v-v'| [ρ_α(v_*) ρ_β(v'_*) - ρ_α(v) ρ_β(v')]

Discretisation: uniform tensor-product grid in (s, v),
Strang operator splitting between streaming (Godunov upwind)
and collision (bilinear-interpolation Boltzmann integral with
Mieussens-style conservative weighting).
=#
module GHDBoltzmannSolver

export GHDBState, GHDBParams, run_simulation!,
       compute_moments, kinetic_entropy,
       streaming_step_notime!, collision_step_notime!,
       strang_step_clean!,
       init_local_maxwell!, init_step_velocity!

using LinearAlgebra

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

struct GHDBParams{T<:AbstractFloat}
    L::T              # arc-length perimeter
    sigma::T          # rod diameter
    m_L::T            # mass of species L
    m_H::T            # mass of species H
    kT::T             # temperature (k_B = 1)
    N_s::Int          # number of spatial cells
    N_v::Int          # number of velocity bins
    V_max::T          # max |v| in velocity grid
    cfl::T            # CFL safety factor
    eta_cap::T        # cap on σρ_tot for v_eff regularisation
end

function GHDBParams(; L::T=2π, sigma::T=0.00785, m_L::T=1.0, m_H::T=5.0,
                     kT::T=1.0, N_s::Int=80, N_v::Int=64,
                     V_max::T=5.0, cfl::T=0.25, eta_cap::T=0.99) where T
    GHDBParams{T}(L, sigma, m_L, m_H, kT, N_s, N_v, V_max, cfl, eta_cap)
end

# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

"""
Collision-kernel cache: precomputed THREE-POINT LAGRANGE interpolation
indices and weights for the inter-species collision integral.

3-pt Lagrange preserves polynomials up to quadratic exactly, so the
discrete collision operator preserves all four collision invariants
(per-species mass, total momentum, total energy) to machine precision,
in contrast to the 2-pt bilinear scheme which preserves only mass and
momentum (energy violated at O(Δv²)).

For each (j, l):
  v_*α^{LH}: pre-collision velocity for species α; stored as central
  index `jc_*` (2 ≤ jc ≤ N_v-1) and three Lagrange weights (wm, w0, wp)
  at indices (jc-1, jc, jc+1). The collision pair is invalid (and
  skipped in BOTH gain and loss to maintain conservation) when either
  v_*L or v_*H falls outside the central-index range.
"""
struct CollisionKernel{T}
    abs_diff::Matrix{T}        # N_v × N_v: |v_j - v_l|
    # LH branch: collision (L of v_j, H of v_l)
    jc_LH_L::Matrix{Int}       # central index for v_*L (L stencil)
    wm_LH_L::Matrix{T}; w0_LH_L::Matrix{T}; wp_LH_L::Matrix{T}
    jc_LH_H::Matrix{Int}
    wm_LH_H::Matrix{T}; w0_LH_H::Matrix{T}; wp_LH_H::Matrix{T}
    valid_LH::Matrix{Bool}
    # HL branch: collision (H of v_j, L of v_l)
    jc_HL_H::Matrix{Int}
    wm_HL_H::Matrix{T}; w0_HL_H::Matrix{T}; wp_HL_H::Matrix{T}
    jc_HL_L::Matrix{Int}
    wm_HL_L::Matrix{T}; w0_HL_L::Matrix{T}; wp_HL_L::Matrix{T}
    valid_HL::Matrix{Bool}
end

@inline function lagrange3_weights(δ::T) where T
    # 3-pt Lagrange basis at offset δ ∈ [-0.5, 0.5] from central node,
    # stencil {-1, 0, +1} spacing 1. Preserves polynomials of degree ≤ 2.
    wm = δ*(δ - T(1))/T(2)
    w0 = T(1) - δ*δ
    wp = δ*(δ + T(1))/T(2)
    return wm, w0, wp
end

@inline function locate3(v_star::T, V_max::T, dv::T, N_v::Int) where T
    # Continuous index (1-based, integer at cell center)
    pos = (v_star + V_max)/dv + T(0.5)
    jc = round(Int, pos)
    if jc < 2 || jc > N_v - 1
        return 0, zero(T)  # signals "out of range"
    end
    δ = pos - jc
    return jc, δ
end

function build_collision_kernel(p::GHDBParams{T}, v::Vector{T}, dv::T) where T
    N_v = p.N_v
    abs_diff = zeros(T, N_v, N_v)
    jc_LH_L = zeros(Int, N_v, N_v)
    wm_LH_L = zeros(T, N_v, N_v); w0_LH_L = zeros(T, N_v, N_v); wp_LH_L = zeros(T, N_v, N_v)
    jc_LH_H = zeros(Int, N_v, N_v)
    wm_LH_H = zeros(T, N_v, N_v); w0_LH_H = zeros(T, N_v, N_v); wp_LH_H = zeros(T, N_v, N_v)
    valid_LH = falses(N_v, N_v)
    jc_HL_H = zeros(Int, N_v, N_v)
    wm_HL_H = zeros(T, N_v, N_v); w0_HL_H = zeros(T, N_v, N_v); wp_HL_H = zeros(T, N_v, N_v)
    jc_HL_L = zeros(Int, N_v, N_v)
    wm_HL_L = zeros(T, N_v, N_v); w0_HL_L = zeros(T, N_v, N_v); wp_HL_L = zeros(T, N_v, N_v)
    valid_HL = falses(N_v, N_v)

    for l in 1:N_v, j in 1:N_v
        abs_diff[j, l] = abs(v[j] - v[l])

        # LH branch: collision of (L at v_j) + (H at v_l)
        # Pre-collision pair (involutive elastic map):
        v_sL, v_sH = elastic_post(v[j], v[l], p.m_L, p.m_H)
        jcL, δL = locate3(v_sL, p.V_max, dv, N_v)
        jcH, δH = locate3(v_sH, p.V_max, dv, N_v)
        if jcL > 0 && jcH > 0
            valid_LH[j, l] = true
            wm, w0, wp = lagrange3_weights(δL)
            jc_LH_L[j, l] = jcL
            wm_LH_L[j, l] = wm; w0_LH_L[j, l] = w0; wp_LH_L[j, l] = wp
            wm, w0, wp = lagrange3_weights(δH)
            jc_LH_H[j, l] = jcH
            wm_LH_H[j, l] = wm; w0_LH_H[j, l] = w0; wp_LH_H[j, l] = wp
        end

        # HL branch: collision of (H at v_j) + (L at v_l)
        v_sHp, v_sLp = elastic_post(v[j], v[l], p.m_H, p.m_L)
        jcH, δH = locate3(v_sHp, p.V_max, dv, N_v)
        jcL, δL = locate3(v_sLp, p.V_max, dv, N_v)
        if jcH > 0 && jcL > 0
            valid_HL[j, l] = true
            wm, w0, wp = lagrange3_weights(δH)
            jc_HL_H[j, l] = jcH
            wm_HL_H[j, l] = wm; w0_HL_H[j, l] = w0; wp_HL_H[j, l] = wp
            wm, w0, wp = lagrange3_weights(δL)
            jc_HL_L[j, l] = jcL
            wm_HL_L[j, l] = wm; w0_HL_L[j, l] = w0; wp_HL_L[j, l] = wp
        end
    end
    return CollisionKernel{T}(abs_diff,
        jc_LH_L, wm_LH_L, w0_LH_L, wp_LH_L,
        jc_LH_H, wm_LH_H, w0_LH_H, wp_LH_H, valid_LH,
        jc_HL_H, wm_HL_H, w0_HL_H, wp_HL_H,
        jc_HL_L, wm_HL_L, w0_HL_L, wp_HL_L, valid_HL)
end

@inline function elastic_post(v1::T, v2::T, m1::T, m2::T) where T
    M = m1 + m2
    v1p = ((m1 - m2)*v1 + 2*m2*v2)/M
    v2p = (2*m1*v1 + (m2 - m1)*v2)/M
    return v1p, v2p
end

mutable struct GHDBState{T}
    p::GHDBParams{T}
    s::Vector{T}       # spatial cell centers, length N_s
    v::Vector{T}       # velocity bin centers, length N_v
    ds::T              # spatial cell width
    dv::T              # velocity bin width
    rho_L::Matrix{T}   # ρ_L[s_k, v_j], size N_s × N_v
    rho_H::Matrix{T}   # ρ_H[s_k, v_j]
    t::T               # current time
    # Pre-allocated scratch arrays (avoid allocation in inner loops)
    rho_L_new::Matrix{T}
    rho_H_new::Matrix{T}
    slope_L::Matrix{T}
    slope_H::Matrix{T}
    V_face::Matrix{T}  # face-centered v_eff per (k, j)
    R_L::Matrix{T}     # collision rate for L
    R_H::Matrix{T}
    # Cached collision kernel
    kernel::CollisionKernel{T}
end

function GHDBState(p::GHDBParams{T}) where T
    ds = p.L / p.N_s
    dv = 2*p.V_max / p.N_v
    s  = T[(k - 0.5)*ds for k in 1:p.N_s]
    v  = T[-p.V_max + (j - 0.5)*dv for j in 1:p.N_v]
    rho_L = zeros(T, p.N_s, p.N_v)
    rho_H = zeros(T, p.N_s, p.N_v)
    rho_L_new = zeros(T, p.N_s, p.N_v)
    rho_H_new = zeros(T, p.N_s, p.N_v)
    slope_L = zeros(T, p.N_s, p.N_v)
    slope_H = zeros(T, p.N_s, p.N_v)
    V_face = zeros(T, p.N_s, p.N_v)
    R_L = zeros(T, p.N_s, p.N_v)
    R_H = zeros(T, p.N_s, p.N_v)
    kernel = build_collision_kernel(p, v, dv)
    GHDBState{T}(p, s, v, ds, dv, rho_L, rho_H, zero(T),
                 rho_L_new, rho_H_new, slope_L, slope_H, V_face,
                 R_L, R_H, kernel)
end

# ---------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------

"""
    init_local_maxwell!(st, n_L, n_H, u_bar, T)

Set ρ_α(s,v) = (n_α(s)/Z_α) exp[-m_α (v - u_bar(s))² / (2 kT)]
with n_α, u_bar, T scalars or arrays of length N_s.
"""
function init_local_maxwell!(st::GHDBState{T}, n_L, n_H, u_bar, Tk) where T
    p = st.p
    for k in 1:p.N_s
        nL_k = n_L isa Number ? n_L : n_L[k]
        nH_k = n_H isa Number ? n_H : n_H[k]
        u_k  = u_bar isa Number ? u_bar : u_bar[k]
        T_k  = Tk isa Number ? Tk : Tk[k]
        Z_L = sqrt(2π*T_k/p.m_L)
        Z_H = sqrt(2π*T_k/p.m_H)
        for j in 1:p.N_v
            dv = st.v[j] - u_k
            st.rho_L[k,j] = (nL_k/Z_L)*exp(-p.m_L*dv^2/(2*T_k))
            st.rho_H[k,j] = (nH_k/Z_H)*exp(-p.m_H*dv^2/(2*T_k))
        end
    end
end

"""
    init_step_velocity!(st, n_L, n_H, U_0, T)

Step velocity IC: u_coh(s) = U_0 sgn(cos(2π s/L)), local Maxwellian
on top of it. Thermal width with temperature T per species.
"""
function init_step_velocity!(st::GHDBState{T}, n_L::T, n_H::T,
                              U_0::T, Tk::T) where T
    p = st.p
    u_bar = T[(cos(2π*st.s[k]/p.L) > 0 ? U_0 : -U_0) for k in 1:p.N_s]
    init_local_maxwell!(st, n_L, n_H, u_bar, Tk)
end

# ---------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------

"""
    compute_moments(st) -> (n_L, n_H, rho_tot, u_bar)

Local hydrodynamic moments at each s-bin. Number-weighted ū per
Paper IV Eq. (10).
"""
function compute_moments(st::GHDBState{T}) where T
    p = st.p
    n_L = vec(sum(st.rho_L, dims=2)) .* st.dv
    n_H = vec(sum(st.rho_H, dims=2)) .* st.dv
    rho_tot = n_L .+ n_H
    sum_v_L = vec(st.rho_L * st.v) .* st.dv
    sum_v_H = vec(st.rho_H * st.v) .* st.dv
    u_bar = (sum_v_L .+ sum_v_H) ./ max.(rho_tot, T(1e-30))
    return n_L, n_H, rho_tot, u_bar
end

"""
    kinetic_entropy(st)

S = -Σ_α ∫ ds dv ρ_α log ρ_α (discretised).
"""
function kinetic_entropy(st::GHDBState{T}) where T
    S = zero(T)
    for k in 1:st.p.N_s, j in 1:st.p.N_v
        for rho in (st.rho_L[k,j], st.rho_H[k,j])
            if rho > T(1e-30)
                S -= rho*log(rho)*st.ds*st.dv
            end
        end
    end
    return S
end

# ---------------------------------------------------------------------
# Streaming step (WENO5 5th-order upwind, conservative)
# ---------------------------------------------------------------------

"""
WENO5 reconstruction of the LEFT state at face k+1/2 from cell averages
(u_{-2}, u_{-1}, u_0, u_{+1}, u_{+2}) indexed relative to cell k.
This is the upwind reconstruction for V > 0 (information flowing left
to right). Returns 5th-order accuracy in smooth regions and ENO-stable
behaviour at discontinuities. ε = 1e-6 regularises smoothness indicators.
"""
@inline function weno5_left(um2::T, um1::T, u0::T, up1::T, up2::T) where T
    ε = T(1e-6)
    # Three candidate reconstructions
    p0 = (T(2)*um2 - T(7)*um1 + T(11)*u0)/T(6)
    p1 = (-um1 + T(5)*u0 + T(2)*up1)/T(6)
    p2 = (T(2)*u0 + T(5)*up1 - up2)/T(6)
    # Smoothness indicators
    β0 = T(13)/T(12)*(um2 - T(2)*um1 + u0)^2 +
         T(1)/T(4)*(um2 - T(4)*um1 + T(3)*u0)^2
    β1 = T(13)/T(12)*(um1 - T(2)*u0 + up1)^2 +
         T(1)/T(4)*(um1 - up1)^2
    β2 = T(13)/T(12)*(u0 - T(2)*up1 + up2)^2 +
         T(1)/T(4)*(T(3)*u0 - T(4)*up1 + up2)^2
    # Nonlinear weights
    α0 = T(1)/T(10) / (ε + β0)^2
    α1 = T(6)/T(10) / (ε + β1)^2
    α2 = T(3)/T(10) / (ε + β2)^2
    αs = α0 + α1 + α2
    return (α0*p0 + α1*p1 + α2*p2)/αs
end

"""
WENO5 reconstruction of the RIGHT state at face k+1/2 from cells
(u_{-1}, u_0, u_{+1}, u_{+2}, u_{+3}) relative to cell k. Mirror of
weno5_left, used for V < 0 (information flowing right to left).
"""
@inline function weno5_right(um1::T, u0::T, up1::T, up2::T, up3::T) where T
    ε = T(1e-6)
    p0 = (-um1 + T(5)*u0 + T(2)*up1)/T(6)
    p1 = (T(2)*u0 + T(5)*up1 - up2)/T(6)
    p2 = (T(11)*up1 - T(7)*up2 + T(2)*up3)/T(6)
    β0 = T(13)/T(12)*(um1 - T(2)*u0 + up1)^2 +
         T(1)/T(4)*(um1 - T(4)*u0 + T(3)*up1)^2
    β1 = T(13)/T(12)*(u0 - T(2)*up1 + up2)^2 +
         T(1)/T(4)*(u0 - up2)^2
    β2 = T(13)/T(12)*(up1 - T(2)*up2 + up3)^2 +
         T(1)/T(4)*(T(3)*up1 - T(4)*up2 + up3)^2
    α0 = T(3)/T(10) / (ε + β0)^2
    α1 = T(6)/T(10) / (ε + β1)^2
    α2 = T(1)/T(10) / (ε + β2)^2
    αs = α0 + α1 + α2
    return (α0*p0 + α1*p1 + α2*p2)/αs
end

"""
    streaming_step_notime!(st, dt)

WENO5 5th-order upwind streaming with face-averaged dressed velocity.
Conservative (flux-difference form). Periodic BCs via mod1 indexing
across the 5-point stencil.
"""
function streaming_step_notime!(st::GHDBState{T}, dt::T) where T
    p = st.p
    n_L, n_H, rho_tot, u_bar = compute_moments(st)
    rho_tot_reg = min.(p.sigma .* rho_tot, p.eta_cap)

    # Face-averaged dressed velocity V_face[k, j] at face k+1/2.
    @inbounds for j in 1:p.N_v, k in 1:p.N_s
        k_p = mod1(k+1, p.N_s)
        u_face = 0.5*(u_bar[k] + u_bar[k_p])
        eta_face = 0.5*(rho_tot_reg[k] + rho_tot_reg[k_p])
        st.V_face[k, j] = (st.v[j] - eta_face*u_face) / (T(1) - eta_face)
    end

    for (rho, rho_new) in ((st.rho_L, st.rho_L_new),
                            (st.rho_H, st.rho_H_new))
        @inbounds Threads.@threads for j in 1:p.N_v
            for k in 1:p.N_s
                # Cell indices relative to k (periodic)
                km2 = mod1(k-2, p.N_s); km1 = mod1(k-1, p.N_s)
                kp1 = mod1(k+1, p.N_s); kp2 = mod1(k+2, p.N_s)
                kp3 = mod1(k+3, p.N_s)
                # Face k+1/2 (right face of cell k)
                Vr = st.V_face[k, j]
                if Vr > 0
                    rho_face_R = weno5_left(rho[km2, j], rho[km1, j],
                                            rho[k,   j], rho[kp1, j],
                                            rho[kp2, j])
                else
                    rho_face_R = weno5_right(rho[km1, j], rho[k,   j],
                                             rho[kp1, j], rho[kp2, j],
                                             rho[kp3, j])
                end
                F_R = Vr * rho_face_R

                # Face k-1/2 (left face of cell k), reuse cell k-1's right face
                Vl = st.V_face[km1, j]
                km3 = mod1(k-3, p.N_s)
                if Vl > 0
                    rho_face_L = weno5_left(rho[km3, j], rho[km2, j],
                                            rho[km1, j], rho[k,   j],
                                            rho[kp1, j])
                else
                    rho_face_L = weno5_right(rho[km2, j], rho[km1, j],
                                             rho[k,   j], rho[kp1, j],
                                             rho[kp2, j])
                end
                F_L = Vl * rho_face_L

                rho_new[k, j] = rho[k, j] - (dt/st.ds)*(F_R - F_L)
            end
        end
        copyto!(rho, rho_new)
    end
end

"""
    collision_step_notime!(st, dt)

Update ρ_L, ρ_H by the inter-species Boltzmann collision integral
(Eq. (17) of Paper V), using 3-point Lagrange interpolation in
velocity space. The discrete operator preserves mass, momentum, and
energy to machine precision (the four collision invariants are exact
null modes of the linearised operator). Off-grid collision pairs are
skipped in BOTH gain and loss to maintain conservation.
"""
function collision_step_notime!(st::GHDBState{T}, dt::T) where T
    p = st.p
    K = st.kernel
    fill!(st.R_L, zero(T))
    fill!(st.R_H, zero(T))

    @inbounds Threads.@threads for k in 1:p.N_s
        @views rho_L_k = st.rho_L[k, :]
        @views rho_H_k = st.rho_H[k, :]
        @views R_L_k = st.R_L[k, :]
        @views R_H_k = st.R_H[k, :]
        for j in 1:p.N_v
            sum_L = zero(T); sum_H = zero(T)
            for l in 1:p.N_v
                # ---- LH branch: collision (L at v_j) + (H at v_l) ----
                # Skip entire pair (gain AND loss) if pre-velocity off-grid.
                if K.valid_LH[j, l]
                    jcL = K.jc_LH_L[j, l]
                    rL_star = K.wm_LH_L[j, l]*rho_L_k[jcL-1] +
                              K.w0_LH_L[j, l]*rho_L_k[jcL]   +
                              K.wp_LH_L[j, l]*rho_L_k[jcL+1]
                    jcH = K.jc_LH_H[j, l]
                    rH_star = K.wm_LH_H[j, l]*rho_H_k[jcH-1] +
                              K.w0_LH_H[j, l]*rho_H_k[jcH]   +
                              K.wp_LH_H[j, l]*rho_H_k[jcH+1]
                    gainL = rL_star * rH_star
                    lossL = rho_L_k[j] * rho_H_k[l]
                    sum_L += K.abs_diff[j, l] * (gainL - lossL)
                end

                # ---- HL branch: collision (H at v_j) + (L at v_l) ----
                if K.valid_HL[j, l]
                    jcH = K.jc_HL_H[j, l]
                    rH_star = K.wm_HL_H[j, l]*rho_H_k[jcH-1] +
                              K.w0_HL_H[j, l]*rho_H_k[jcH]   +
                              K.wp_HL_H[j, l]*rho_H_k[jcH+1]
                    jcL = K.jc_HL_L[j, l]
                    rL_star = K.wm_HL_L[j, l]*rho_L_k[jcL-1] +
                              K.w0_HL_L[j, l]*rho_L_k[jcL]   +
                              K.wp_HL_L[j, l]*rho_L_k[jcL+1]
                    gainH = rH_star * rL_star
                    lossH = rho_H_k[j] * rho_L_k[l]
                    sum_H += K.abs_diff[j, l] * (gainH - lossH)
                end
            end
            R_L_k[j] = p.sigma * st.dv * sum_L
            R_H_k[j] = p.sigma * st.dv * sum_H
        end
    end
    @. st.rho_L += dt * st.R_L
    @. st.rho_H += dt * st.R_H
end

function strang_step_clean!(st::GHDBState{T}, dt::T) where T
    half_dt = dt/2
    collision_step_notime!(st, half_dt)
    streaming_step_notime!(st, dt)
    collision_step_notime!(st, half_dt)
    st.t += dt
end

# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

"""
    run_simulation!(st, T_max; dt=auto, save_every=Inf, callback=nothing)

Evolve the state from st.t to st.t + T_max using Strang splitting.
Returns vector of (t, callback_result) at save times if callback given.
"""
function run_simulation!(st::GHDBState{T}, T_max::T;
                         dt::Union{Nothing,T}=nothing,
                         save_every::T=T(Inf),
                         callback=nothing) where T
    p = st.p
    # CFL: streaming and collision constraints. Use ACTUAL state to estimate
    # V_eff_max rather than the worst-case eta_cap, which is far too pessimistic.
    _, _, rho_tot_init, u_bar_init = compute_moments(st)
    eta_max = max(maximum(p.sigma .* rho_tot_init), T(0.01))
    V_eff_max = (p.V_max + p.sigma*maximum(abs, u_bar_init)) / (1 - eta_max)
    dt_stream = p.cfl * st.ds / V_eff_max
    # Approximate τ_break:
    rho_0 = (T(1.0) + T(1.0))/p.L  # placeholder; will be refined
    n_L, n_H, _, _ = compute_moments(st)
    rho_0 = (sum(n_L) + sum(n_H)) / (p.N_s)
    v_bar = sqrt(p.kT*(p.m_L + p.m_H)/(p.m_L*p.m_H))
    tau_break = T(1) / (rho_0 * p.sigma * v_bar + T(1e-12))
    dt_coll = p.cfl * tau_break
    dt_actual = dt === nothing ? min(dt_stream, dt_coll) : dt

    records = Any[]
    t_end = st.t + T_max
    t_next_save = st.t + save_every
    if callback !== nothing
        push!(records, (st.t, callback(st)))
    end
    n_steps = 0
    while st.t < t_end - T(1e-12)
        h = min(dt_actual, t_end - st.t)
        strang_step_clean!(st, h)
        n_steps += 1
        if callback !== nothing && st.t >= t_next_save
            push!(records, (st.t, callback(st)))
            t_next_save += save_every
        end
    end
    return records, n_steps, dt_actual
end

end # module
