# Event-driven hard-rod simulation with reaction-time delay

Design spec for the numerical test of Paper VI Sec. VI
(τ_r^c shock-onset threshold).

## Baseline algorithm (P3 reference)

Standard event-driven hard-rod integrator:

```
state: positions s_i ∈ [0, L), velocities v_i, masses m_i
event queue: (t_event, i, j) pairs for next collision of each pair

main loop:
  pop next event (t_c, i, j)
  advance all particles: s_k ← s_k + v_k · (t_c - t_now)
  apply elastic collision rule to (i, j):
    v_i', v_j' from Eq. (4) of PaperIV
  recompute next events for i and j with all neighbors
  push new events
```

## Delay extension

The reaction-time delay τ_r is implemented as a deferred velocity
update. Each rod carries an additional field `t_release_i`: the time
at which its post-collision velocity will take effect.

```
state: s_i, v_i, m_i, t_release_i (initially -∞)
       v_pending_i (the velocity awaiting release, or NaN if none)
```

### Modified collision handling

```
on collision event (t_c, i, j):
  advance all particles to t_c
  IF t_now < t_release_i OR t_now < t_release_j:
    # one or both is still "frozen" on its pre-collision trajectory
    # Treat as if collision happens but is queued
    pass-through: rods overlap geometrically during freeze
    (allow this only for the duration of the freeze — the
     event handler clamps the overlap or skips it)
  ELSE:
    compute v_i', v_j' from elastic collision rule
    v_pending_i ← v_i';  v_pending_j ← v_j'
    t_release_i ← t_c + τ_r;  t_release_j ← t_c + τ_r
    # but rods continue with OLD velocities until release
  recompute next events with current v_i, v_j

new event type: RELEASE event at (t_release_i, i):
  v_i ← v_pending_i
  v_pending_i ← NaN; t_release_i ← -∞
  recompute next collision events for i
```

### Key subtleties

1. **Geometric overlap during freeze.** With τ_r > 0, rods can
   geometrically overlap (one continues with pre-collision velocity
   into a neighbor's space). For τ_r small compared to inter-collision
   time τ_break, this overlap is brief. For τ_r ~ τ_break the
   overlaps stack and the dynamics degenerate.

   Decision: allow overlap, treat each rod independently during its
   freeze. Multi-collision pile-ups are computed as a sequence of
   pairwise events with their own t_release.

2. **Cascade collisions during freeze.** Rod i with t_release_i in
   the future may experience another "collision" with k before
   release. In that case the colliding velocities are the CURRENT
   ones (pre-release), not the pending ones. The new collision
   replaces the pending velocity:
   ```
   v_pending_i ← (new collision result with k)
   t_release_i ← t_c_new + τ_r  (resets the timer)
   ```

3. **Energy conservation.** Energy is NOT conserved instantaneously
   when v changes at release (the rod jumps from old v to v_pending).
   Over an ensemble average it should still be conserved because
   the collision rule is elastic; the freeze only delays the swap.

4. **Event-queue invalidation.** When a release event fires, all
   pending collision events for that rod must be recomputed with
   the new velocity. This is the standard event-queue invalidation
   pattern.

## Sweep protocol (Sec VI of P6)

Parameters from Ref. PaperIII:
- L = 12.57, σ = 0.00314, ρ_tot = 31.82 (N = 400 rods)
- η = 0.10, m_L = 1.0, m_H = 5.0 (equimolar)
- T = 1 (kinetic temperature)
- Step IC: u(s, 0) = U_0 sgn(cos(2π s / L)) with U_0 = 0.5
- Maxwell-Boltzmann velocity sampling at each s

Delay sweep:
- τ_r ∈ {0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0}
- Run T_sim = 12 T_sound ≈ 135 time units
- Save trajectories at save_dt = 0.1
- 20-50 realisations per τ_r (different RNG seeds)

Observables:
- |ρ̂_n(t)| for n = 1, ..., 7 (FFT of binned density)
- R^{e/o}(t) = |ρ̂_2(t)| / |ρ̂_3(t)|
- First-peak R^{e/o} = R^{e/o}(t_peak) where t_peak = argmax_t |ρ̂_1(t)|

Diagnostic:
- τ_r^c is the smallest τ_r for which ⟨R^{e/o}⟩ > 0.5
- Compare to analytical prediction τ_r^c ≈ 2.8 from Eq. (eq:tauc)

## Estimated runtime

Event-driven N=400 hard rods, T_sim = 135 units, ~20 realisations:
- baseline (τ_r = 0): O(N² log N · n_events) ~ 1-10 min per run
- with delay: ~1.5x baseline (extra release events, queue churn)
- Full sweep: 7 × 20 × ~5 min = ~12 hours wall time

Tractable on a single workstation.

## Implementation status

The P3 event-driven code is in a separate Julia project not present
in this directory. The delay extension above is implementable on
top of any standard event-driven hard-rod integrator
(e.g., Allen-Tildesley style with priority-queue event scheduling).

Next step: locate or rebuild the P3 integrator, add the
t_release_i / v_pending_i fields, implement the release event type,
run the sweep.
