"""Numpy mirror of the Seren SDE shake (no torch needed).

Extends the Seren mirror (tests/test_seren.py) with the
error-gated shake: after each accepted step, gamma =
min(err/tol, 1) * sde_strength of the state's implied noise is
swapped for fresh noise along the RF interpolation line
x = (1-s)*x0 + s*eps. (An earlier draft blended the raw residual
(x - x0)/s as unit noise and diverged on this field - final error
556 - which is why the formula is pinned to the interpolation
line here and in nodes.py.)

Same ODE / analytic solution / constants as test_seren.py.

Checks:
1. sde_strength = 0 reproduces the deterministic walk exactly.
2. With strength > 0, different noise seeds give different final
   states (diversity) and the walk still completes in budget.

The error bound is deliberately loose. This test field is
expansive toward sigma=0 (deviations grow like e^{lambda*d}) and
its "denoised" is a linear extrapolation, not a trained
denoiser's contractive pull back to the manifold, so ANY
stochasticity is punished here far beyond what a real model
does. The mirror guards against explosion, not against the
endpoint moving - moving the endpoint is the sampler's purpose.
Measured on this field: strength 0.1/0.2/0.3/0.5 -> final max
err ~3/6/10/13 vs 0.083 deterministic, roughly linear in
strength. Visual judgment on a real model is the real test.
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


def walk(x, tol, beta, max_steps, sde_strength=0.0, noise_seed=0):
    rng = np.random.default_rng(noise_seed)
    stats = {"accepted": 0, "trials": 0, "evals": 0, "shake_sum": 0.0}
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
            if sde_strength > 0.0:
                gamma = min(err / tol, 1.0) * float(sde_strength)
                if gamma > 0.0:
                    d = x_trap - s_next * v_pred
                    eps = rng.normal(size=x.shape)
                    keep = (1.0 - gamma * gamma) ** 0.5
                    eps_implied = (x_trap - (1.0 - s_next) * d) / s_next
                    x = ((1.0 - s_next) * d
                         + s_next * (keep * eps_implied + gamma * eps))
                    stats["shake_sum"] += gamma
            v = v_pred
            m = beta * m + (1.0 - beta) * v_pred
            t = t_next
            sigma = s_next
            stats["accepted"] += 1
            dt = dt_trial * min(max(factor, 0.2), GROW_MAX)
        else:
            dt = dt_trial * min(max(factor, SHRINK_MIN), 0.95)
    return x, stats


def analytic_final(x_start):
    xp1 = P * np.sin(2.0) + Q * np.cos(2.0)
    c = (x_start - xp1) * np.exp(LAMBDA)
    return Q + c


def main():
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=(2, 4, 8, 8)) * 0.7
    exact = analytic_final(x0)

    det, _ = walk(x0.copy(), 0.005, 0.5, 120, sde_strength=0.0)
    zero, _ = walk(x0.copy(), 0.005, 0.5, 120, sde_strength=0.0,
                   noise_seed=99)
    assert np.array_equal(det, zero), "strength 0 must be deterministic"
    det_err = float(np.max(np.abs(det - exact)))
    print(f"deterministic: final max err={det_err:.5f}")

    finals = []
    for seed in (1, 2, 3, 4):
        xf, st = walk(x0.copy(), 0.005, 0.5, 120, sde_strength=0.3,
                      noise_seed=seed)
        err = float(np.max(np.abs(xf - exact)))
        mean_shake = st["shake_sum"] / max(st["accepted"], 1)
        finals.append(xf)
        print(f"seed={seed}: accepted={st['accepted']}, "
              f"evals={st['evals']}, mean_shake={mean_shake:.3f}, "
              f"final max err={err:.5f}")
        assert np.all(np.isfinite(xf))
        assert err < 25.0, "shake must not explode even on this field"
        assert err > det_err, "shaken endpoint must actually move"
        assert 0.0 < mean_shake < 0.3
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            assert not np.allclose(finals[i], finals[j]), \
                "different noise seeds must diverge"
    print("diversity confirmed across noise seeds")
    print("all checks passed")


if __name__ == "__main__":
    main()
