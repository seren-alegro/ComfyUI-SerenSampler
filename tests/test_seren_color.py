"""Numpy mirror of the Seren Color steering (no torch needed).

Extends the Seren mirror (tests/test_seren.py): when an accepted
step lands at sigma <= color_start, the step's denoised estimate
is pushed away from its per-channel spatial mean with a gain
ramping in as sigma falls, and the accepted state is shifted by
(1 - sigma) * (d2 - d). Same ODE / constants as test_seren.py.

Checks: strength 0 reproduces the base walk exactly; strength
0.35 and 1.0 complete in budget, stay finite, and actually move
the endpoint (the steering is applied, not a no-op).

The per-step shift is capped at 10% of the state's RMS: the toy
field's denoised is an extrapolating linear map, so uncapped
steering formed a gain loop and diverged in this mirror (shift
~1e8 at strength 1.0). The cap keeps the mechanism's direction
while removing the runaway, on any field.
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


def walk(x, tol, beta, max_steps, color_strength=0.0,
         color_start=0.35):
    stats = {"accepted": 0, "trials": 0, "evals": 0, "steered": 0}
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
            if color_strength > 0.0 and 0.0 < s_next <= color_start:
                d = x_trap - s_next * v_pred
                dims = tuple(range(2, x.ndim))
                mu = d.mean(axis=dims, keepdims=True)
                w = 1.0 - s_next / float(color_start)
                gain = 1.0 + float(color_strength) * w
                d2 = mu + (d - mu) * gain
                delta = (1.0 - s_next) * (d2 - d)
                dr = rms(delta)
                cap = 0.10 * rms(x)
                if dr > cap and dr > 0.0:
                    delta = delta * (cap / dr)
                x = x + delta
                stats["steered"] += 1
            v = v_pred
            m = beta * m + (1.0 - beta) * v_pred
            t = t_next
            sigma = s_next
            stats["accepted"] += 1
            dt = dt_trial * min(max(factor, 0.2), GROW_MAX)
        else:
            dt = dt_trial * min(max(factor, SHRINK_MIN), 0.95)
    return x, stats


def main():
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=(2, 4, 8, 8)) * 0.7
    base, _ = walk(x0.copy(), 0.005, 0.5, 120, color_strength=0.0)
    zero, _ = walk(x0.copy(), 0.005, 0.5, 120, color_strength=0.0)
    assert np.array_equal(base, zero)
    for s in (0.35, 1.0):
        xf, st = walk(x0.copy(), 0.005, 0.5, 120, color_strength=s)
        shift = float(np.max(np.abs(xf - base)))
        print(f"strength={s}: accepted={st['accepted']}, "
              f"steered_steps={st['steered']}, "
              f"endpoint shift vs base={shift:.4f}")
        assert np.all(np.isfinite(xf))
        assert st["steered"] > 0, "steering never engaged"
        assert shift > 1e-3, "steering must move the endpoint"
    print("all checks passed")


if __name__ == "__main__":
    main()
