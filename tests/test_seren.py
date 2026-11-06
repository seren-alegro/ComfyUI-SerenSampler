"""Numpy mirror of the Seren sampler core (no torch needed).

Same test field as the AdaptiveFlow mirror: the linear ODE
dx/dsigma = -LAMBDA * (x - sin(2*sigma)), LAMBDA = 1.5, with the
closed-form solution x_p = P sin(2s) + Q cos(2s), P = 0.36,
Q = -0.48 for the particular part.

The mirror re-implements the node's walk with the same constants
and structure (momentum predictor, one evaluation per trial,
trapezoid correction, prediction-vs-correction error, trial budget
and the fixed-Euler fallback), in velocity space (the node reaches
velocities through the denoised prediction; the mirror calls the
field directly).

Checks:
1. Tighter tolerance -> more accepted steps, smaller final error.
2. Evaluation economy: at a mid tolerance the total model
   evaluations stay well below 2 per accepted step (the embedded
   pair scheme needs ~2), because prediction is free.
3. beta = 0, 0.5, 0.9 all complete with finite, sane results.
4. An unreachable tolerance with a tiny budget still completes
   (fallback) with a finite state.
"""
import numpy as np

SAFETY = 0.9
GROW_MAX = 2.5
SHRINK_MIN = 0.1
MIN_DT = 1e-4
SHIFT = 3.0
LAMBDA = 1.5
P = 0.36
Q = -0.48


def shift_sigma(t):
    return SHIFT * t / (1.0 + (SHIFT - 1.0) * t)


def velocity(x, sigma):
    return -LAMBDA * (x - np.sin(2.0 * sigma))


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))


def walk(x, tol, beta, max_steps):
    stats = {"accepted": 0, "trials": 0, "evals": 0}
    t = 1.0
    sigma = 1.0
    max_trials = 2 * int(max_steps) + 16
    v = velocity(x, sigma)
    stats["evals"] += 1
    m = v.copy()
    dt = -(t / 30.0)
    while t > 0.0:
        if stats["trials"] >= max_trials:
            break
        dt_trial = dt
        if t + dt_trial < 0.0:
            dt_trial = -t
        t_next = t + dt_trial
        s_next = shift_sigma(t_next) if t_next > 0.0 else 0.0
        stats["trials"] += 1
        if s_next == 0.0:
            stats["evals"] += 1
            x = x - sigma * velocity(x, sigma)  # denoised
            t = 0.0
            stats["accepted"] += 1
            break
        x_pred = x + m * (s_next - sigma)
        v_pred = velocity(x_pred, s_next)
        stats["evals"] += 1
        x_trap = x + 0.5 * (v + v_pred) * (s_next - sigma)
        err = rms(x_trap - x_pred) / (rms(x_trap) + 1e-6)
        factor = SAFETY * (tol / max(err, 1e-12)) ** 0.5
        if err <= tol or abs(dt_trial) <= MIN_DT:
            x = x_trap
            v = v_pred
            m = beta * m + (1.0 - beta) * v_pred
            t = t_next
            sigma = s_next
            stats["accepted"] += 1
            dt = dt_trial * min(max(factor, 0.2), GROW_MAX)
        else:
            dt = dt_trial * min(max(factor, SHRINK_MIN), 0.95)
    if t > 0.0:  # fixed-Euler fallback, mirrors the node
        remaining = max(1, int(round(t * 30)))
        grid_t = [t * (1.0 - k / remaining) for k in range(remaining + 1)]
        for k in range(remaining):
            s_cur = shift_sigma(grid_t[k])
            s_nxt = shift_sigma(grid_t[k + 1]) if grid_t[k + 1] > 0.0 else 0.0
            stats["evals"] += 1
            if s_nxt == 0.0:
                x = x - s_cur * velocity(x, s_cur)
            else:
                x = x + velocity(x, s_cur) * (s_nxt - s_cur)
            stats["accepted"] += 1
    return x, stats


def analytic_final(x_start):
    xp1 = P * np.sin(2.0) + Q * np.cos(2.0)
    c = (x_start - xp1) * np.exp(LAMBDA)
    return Q + c


def main():
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=(2, 4, 8, 8)) * 0.7
    exact = analytic_final(x0)

    results = {}
    for tol in (0.05, 0.005, 0.0005):
        xf, st = walk(x0.copy(), tol, beta=0.5, max_steps=120)
        err = float(np.max(np.abs(xf - exact)))
        results[tol] = (st, err)
        print(f"tol={tol:<7} accepted={st['accepted']:<4} "
              f"trials={st['trials']:<4} evals={st['evals']:<4} "
              f"final max err={err:.5f}")

    counts = [results[t][0]["accepted"] for t in (0.05, 0.005, 0.0005)]
    errs = [results[t][1] for t in (0.05, 0.005, 0.0005)]
    assert counts[0] < counts[1] < counts[2], "step counts must grow"
    assert errs[0] > errs[1] > errs[2], "errors must shrink"

    st_mid = results[0.005][0]
    assert st_mid["evals"] < 1.6 * st_mid["accepted"] + 4, \
        "prediction must keep evaluations near 1 per step"

    for beta in (0.0, 0.9):
        xf, st = walk(x0.copy(), 0.005, beta=beta, max_steps=120)
        err = float(np.max(np.abs(xf - exact)))
        assert np.all(np.isfinite(xf)) and err < 0.2, (beta, err)
        print(f"beta={beta}: accepted={st['accepted']}, "
              f"evals={st['evals']}, final max err={err:.5f}")

    xf, st = walk(x0.copy(), 1e-9, beta=0.5, max_steps=6)
    assert np.all(np.isfinite(xf)), "fallback must stay finite"
    print(f"fallback run: accepted={st['accepted']}, evals={st['evals']}, "
          "completed at t=0")
    print("all checks passed")


if __name__ == "__main__":
    main()
