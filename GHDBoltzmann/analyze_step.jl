#=
Post-hoc analysis of long_step_sim.jl output. The "peak over all
time" measurement is contaminated by mode-dependent damping
(γ_n ~ n²k_1² D gives faster decay for higher n). The proper
linear-regime |ρ̂_n|/|ρ̂_1| ratio Eq.~(ratio) of P4 is the FIRST
peak of each mode.
=#
using DelimitedFiles, Printf, Statistics

const T_sound = 11.31
const c_s_an = 1.111
const k1 = 2π/12.57

function analyze(fname)
    data = readdlm(joinpath(@__DIR__, fname), ',', skipstart=1)
    t = data[:, 1]
    rho = data[:, 2:8]  # n=1..7

    println("="^60)
    println("File: $fname")
    println("="^60)

    # First peak for each mode (local max in the first T_sound interval after t=0)
    println("\nFIRST PEAK of each mode (within first $(round(T_sound, digits=1)) time units):")
    println("n | t_peak | |ρ̂_n|_first | ratio/n=1 | 1/n | dev (%)")
    println("---|---|---|---|---|---")
    peaks = zeros(7)
    times_peak = zeros(7)
    for n in 1:7
        col = rho[:, n]
        # Find first local maximum
        i_peak = 0
        for i in 2:length(col)-1
            if t[i] > T_sound  # only look in first period
                break
            end
            if col[i] > col[i-1] && col[i] > col[i+1]
                i_peak = i
                break
            end
        end
        if i_peak == 0
            # fallback: max over first period
            mask = t .< T_sound
            i_peak = argmax(col[mask])
        end
        peaks[n] = col[i_peak]
        times_peak[n] = t[i_peak]
    end
    for n in 1:7
        ratio = peaks[n]/peaks[1]
        inv_n = 1.0/n
        dev = isodd(n) ? (ratio - inv_n)/inv_n * 100 : NaN
        if isnan(dev)
            @printf("%d | %.2f | %.4e | %.4f | (even) | --- \n",
                    n, times_peak[n], peaks[n], ratio)
        else
            @printf("%d | %.2f | %.4e | %.4f | %.4f | %+.1f%% \n",
                    n, times_peak[n], peaks[n], ratio, inv_n, dev)
        end
    end

    # Damping: from successive maxima of n=1 mode
    rho1 = rho[:, 1]
    maxima_t = Float64[]; maxima_v = Float64[]
    for i in 2:length(rho1)-1
        if rho1[i] > rho1[i-1] && rho1[i] > rho1[i+1] && t[i] > T_sound/4
            push!(maxima_t, t[i]); push!(maxima_v, rho1[i])
        end
    end
    if length(maxima_t) >= 3
        log_v = log.(maxima_v)
        tc = maxima_t .- mean(maxima_t)
        lc = log_v .- mean(log_v)
        γ = -sum(tc.*lc)/sum(tc.^2)
        @printf("\nDamping γ from %d n=1 maxima: γ = %.5f\n", length(maxima_t), γ)
        @printf("  Q = ω/(2γ) = %.2f\n", c_s_an*k1/(2*γ))
        @printf("  D_apparent = γ/k1² = %.4f\n", γ/k1^2)
    else
        println("\nOnly $(length(maxima_t)) maxima for damping fit")
    end

    return peaks, times_peak
end

analyze("step_sim_eigenmode_A05.csv")
analyze("step_sim_step_U005.csv")
analyze("step_sim_step_U05.csv")
