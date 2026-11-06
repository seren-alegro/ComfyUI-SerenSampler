"""Experiment: does periodic verification help the Seren walk?

Variant of the Seren mirror (tests/test_seren.py). Every K accepted
steps, the walk spends ONE extra evaluation at the current point:
it measures the drift between the carried velocity (evaluated at
the previous step's predicted point) and a fresh velocity at the
actual corrected point, then re-anchors (carried velocity and
momentum are refreshed from the fresh value).

Question: at the same tolerance, does the final error drop enough
to pay for the extra evaluations? Same ODE / analytic solution /
constants as test_seren.py.
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


def walk(x, tol, beta, max_steps, verify_every=0):
    stats = {"accepted": 0, "trials": 0, "evals": 0, "verifies": 0}
    drifts = []
    t = 1.0
    sigma = 1.0
    max_trials = 2 * int(max_steps) + 16
    v = velocity(x, sigma)
    stats["evals"] += 1
    m = v.copy()
    dt = -(t / 30.0)
    since_verify = 0
    while t > 0.0:
        if stats["trials"] >= max_trials:
            break
        if (verify_every and since_verify >= verify_every
                and stats["accepted"] > 0):
            v_fresh = velocity(x, sigma)
            stats["evals"] += 1
            stats["verifies"] += 1
            drifts.append(rms(v_fresh - v) / (rms(v_fresh) + 1e-6))
            v = v_fresh
            m = beta * m + (1.0 - beta) * v_fresh
            since_verify = 0
        dt_trial = dt
        if t + dt_trial < 0.0:
            dt_trial = -t
        t_next = t + dt_trial
        s_next = shift_sigma(t_next) if t_next > 0.0 else 0.0
        stats["trials"] += 1
        if s_next == 0.0:
            stats["evals"] += 1
            x = x - sigma * velocity(x, sigma)
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
            since_verify += 1
            dt = dt_trial * min(max(factor, 0.2), GROW_MAX)
        else:
            dt = dt_trial * min(max(factor, SHRINK_MIN), 0.95)
    return x, stats, drifts


def analytic_final(x_start):
    xp1 = P * np.sin(2.0) + Q * np.cos(2.0)
    c = (x_start - xp1) * np.exp(LAMBDA)
    return Q + c


def main():
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=(2, 4, 8, 8)) * 0.7
    exact = analytic_final(x0)
    for tol in (0.005, 0.0005):
        print(f"--- tolerance {tol} ---")
        for k in (0, 8, 4):
            xf, st, drifts = walk(x0.copy(), tol, beta=0.5,
                                  max_steps=200, verify_every=k)
            err = float(np.max(np.abs(xf - exact)))
            label = "no verify" if k == 0 else f"verify /{k}"
            drift_note = ""
            if drifts:
                drift_note = (f" | drift mean {np.mean(drifts):.4f}, "
                              f"max {np.max(drifts):.4f}")
            print(f"{label:<11} accepted={st['accepted']:<4} "
                  f"evals={st['evals']:<4} verifies={st['verifies']:<3} "
                  f"final max err={err:.5f}{drift_note}")


if __name__ == "__main__":
    main()
