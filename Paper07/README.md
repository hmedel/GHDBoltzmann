# Paper VII — Shock onset in non-integrable hard-rod gases

**Titulo:** "Shock onset in non-integrable hard-rod gases:
inelasticity, non-reciprocity, and delay"

**Target:** PRE o J. Stat. Mech.

## Motivacion

P6 demostro que el gas integrable de varillas duras resiste la formacion
de choques inducidos por delay (sigma_dec ~ 30 >> t*_shock^{-1} ~ 0.25).
La integrabilidad blinda al sistema: los solitones GHD decorrelacionan
la perturbacion antes de que la cascada no lineal forme un choque.

P7 rompe sistematicamente la integrabilidad por tres canales para encontrar
el umbral que P6 predijo pero no pudo realizar.

## Tres canales de ruptura de integrabilidad

| Canal | Parametro | Fisica | Efecto esperado |
|-------|-----------|--------|-----------------|
| (a) Inelasticidad | epsilon > 0 (r < 1) | Colisiones drenan KE coherente | Amortigua solitones, reduce sigma_dec |
| (b) OVM no reciproco | alpha > 0 | Anticipacion solo hacia adelante: dv_i/dt = alpha[V_opt(g_i^fwd(t-tau)) - v_i] | Rompe balance detallado y conservacion de momento |
| (c) Delay | tau_r > 0 | Tiempo de reaccion (de P6) | Fuente de inestabilidad |

## Pregunta central

Cual combinacion (epsilon, alpha, tau_r) produce un crossover monotonico
en R_2(tau_r)? Mapear el espacio de fases (epsilon, alpha, tau_r) para
encontrar donde el umbral del continuo tau_r^c se materializa microscopicamente.

## Infraestructura

- Reutiliza scripts de P6 (framework run_p6_ovm_scan.jl)
- Extiende con regla de colision inelastica epsilon > 0 de P4
- Mismos parametros base: N=400, eta=0.10, masas binarias m_L=1, m_H=5, step IC U_0=0.5

## Plan de simulaciones

1. **Scan (a):** epsilon in {0.01, 0.05, 0.10, 0.20} con tau_r sweep, alpha=0
2. **Scan (b):** alpha in {0.1, 0.5, 1.0, 2.0} con tau_r sweep, epsilon=0
3. **Scan (a+b+c):** combinaciones en el espacio (epsilon, alpha, tau_r)
4. **Diagnosticos:** R_2(tau_r), sigma_dec(epsilon, alpha), espectro de modos

## Motivacion fisica

El trafico real tiene los tres ingredientes: perdida inelastica de energia
(frenado), sensado no reciproco (solo hacia adelante), y delay de reaccion.
P7 conecta la serie integrable de varillas duras con trafico realista
activando estos canales uno a uno y en combinacion.

## Referencias clave

- Paper VI (este proyecto) — delay en gas integrable, sigma_dec >> t*_shock^{-1}
- Paper III — varillas con desorden de masa resisten formacion de choques de Burgers
- Paper IV — GHD con desorden binario de masa
- Sugiyama et al (2008) — embotellamiento en anillo
- Bando et al (1995) — modelo OVM original
