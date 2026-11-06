"""Seren Sampler - momentum-predicted, tolerance-controlled flow.

An experimental sampler for Anima-family rectified flow that fuses
the two ideas from the sibling packs (FlowMomentum's velocity EMA
and AdaptiveFlow's tolerance control) into one mechanism:

- A momentum vector (exponential moving average of past velocities)
  PREDICTS the next point for free: x_pred = x + m * dSigma.
- One model evaluation at the predicted point gives v_pred; a
  trapezoid correction using the carried velocity and v_pred gives
  the candidate x_trap.
- The gap between prediction and correction IS the error estimate.
  Inside tolerance: accept, fold v_pred into the momentum, grow the
  step. Outside: shrink and retry. Typical cost is ONE model
  evaluation per accepted step.

How this is not other momentum samplers: it is a deterministic
rectified-flow ODE walk (no stochastic noise term, no look-ahead
model evaluation at a momentum-simulated point blended back, no
adaptive noise). The momentum never touches the model input - it
only proposes points and measures surprise. Step count is an
outcome of the tolerance, as in AdaptiveFlow.

Safety valves mirror AdaptiveFlow: a minimum step floor (accept
as-is below it) and a trial budget (2 x max_steps + 16 trials); if
the budget runs out mid-flight the remainder is finished with
plain fixed-step Euler so a complete sample always comes out.

The KSampler dropdown registration ("seren") runs the same core
with fixed defaults (tolerance 0.01, beta 0.5); its shift is
estimated from the incoming sigma grid. The standalone node is the
way to tune the knobs.

The second sampler, Seren SDE, is the stochastic sibling: after
each accepted step it shakes the state with fresh noise whose
amount is proportional to how hard that step was (the accepted
step's err / tolerance ratio, scaled by sde_strength). The shake
replaces part of the state's implied noise along the rectified
flow interpolation line (x = (1-s)*x0 + s*eps), so the state
stays on the flow manifold; calm stretches stay almost
deterministic and uncertain stretches get the randomness. This
is not a direction-confidence noise schedule - the noise follows
the local error topography of this specific walk, and the
step-size controller still governs the walk.

The third sampler, Seren Color, steers colour during the walk
instead of boosting it afterwards (how chroma-boost sampler
variants usually work): in the low-sigma stretch where colour
settles, each accepted step's denoised estimate is pushed away
from its own per-channel spatial mean (colourfulness and feature
contrast up) and the accepted state is shifted to match, so the
trajectory itself bends toward richer colour. Above
color_start sigma it is exactly the base Seren walk.

Independent implementation, written for this pack.
"""

import torch

import comfy.sample
import comfy.samplers

SAFETY = 0.9          # controller safety factor
GROW_MAX = 2.5        # max step growth per accepted step
SHRINK_MIN = 0.1      # max shrink per rejected step
MIN_DT = 1e-4         # t-space step floor: accept as-is below this
DEFAULT_TOLERANCE = 0.01
DEFAULT_BETA = 0.5


def shift_sigma(t, shift):
    """sigma for parameter t on the linear-shift schedule."""
    s = float(shift)
    return s * t / (1.0 + (s - 1.0) * t)


def shift_t(sigma, shift):
    """Inverse of shift_sigma."""
    s = float(shift)
    sig = float(sigma)
    return sig / (s - (s - 1.0) * sig)


def estimate_shift(sigmas):
    """Estimate the linear-shift parameter of an incoming grid from
    its midpoint value (sigma at t = 0.5 is shift / (1 + shift)).
    Falls back to 3.0 when the grid does not look like one."""
    try:
        n = len(sigmas)
        if n < 3:
            return 3.0
        mid = float(sigmas[n // 2])
        if 0.05 < mid < 0.98:
            return mid / (1.0 - mid)
    except Exception:
        pass
    return 3.0


def flow_linear_shift_sigmas(steps, shift, denoise=1.0):
    """Nominal grid handed to sample_custom (progress total only);
    the seren walk picks its own steps."""
    t = torch.linspace(float(denoise), 0.0, int(steps) + 1,
                       dtype=torch.float32)
    s = float(shift)
    sigmas = s * t / (1.0 + (s - 1.0) * t)
    sigmas[-1] = 0.0
    return sigmas


def _model_wants_temporal(model):
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    mod = type(dm).__module__ if dm is not None else ""
    return "cosmos" in mod or "anima" in mod


def _fix_latent(model, latent_image):
    """Channel-fix the samples tensor; returns (samples, expanded)."""
    samples = comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"])
    expanded = _model_wants_temporal(model) and samples.ndim == 4
    if expanded:
        samples = samples.unsqueeze(2)
    return samples, expanded


def _rms(t):
    return float(torch.sqrt(torch.mean(t.float() ** 2)))


def _denoised(model, x, sigma, extra_args, stats):
    stats["evals"] += 1
    return model(x, sigma * x.new_ones([x.shape[0]]), **extra_args)


def _seren_loop(model, x, extra_args, callback, tolerance, beta,
                shift, max_steps, stats, t_start=1.0,
                sde_strength=0.0, noise_seed=0,
                color_strength=0.0, color_start=0.35):
    """Momentum-predicted, tolerance-controlled walk to sigma 0.

    Carried state: v (velocity at the current point, evaluated when
    that point was the predicted endpoint of the previous step) and
    m (EMA of accepted endpoint velocities).
    """
    tol = float(tolerance)
    b = float(beta)
    t = float(t_start)
    sigma = shift_sigma(t, shift)
    max_trials = 2 * int(max_steps) + 16

    denoised = _denoised(model, x, sigma, extra_args, stats)
    v = (x - denoised) / sigma
    m = v.clone()
    dt = -(t / 30.0)  # first trial: one nominal 30-step interval

    while t > 0.0:
        if stats["trials"] >= max_trials:
            break
        dt_trial = dt
        if t + dt_trial < 0.0:
            dt_trial = -t
        t_next = t + dt_trial
        s_next = shift_sigma(t_next, shift) if t_next > 0.0 else 0.0
        stats["trials"] += 1
        if s_next == 0.0:
            # Final step: one fresh evaluation, return its x0.
            denoised = _denoised(model, x, sigma, extra_args, stats)
            x = denoised
            t = 0.0
            stats["accepted"] += 1
            break
        x_pred = x + m * (s_next - sigma)
        denoised_p = _denoised(model, x_pred, s_next, extra_args, stats)
        v_pred = (x_pred - denoised_p) / s_next
        x_trap = x + 0.5 * (v + v_pred) * (s_next - sigma)
        err = _rms(x_trap - x_pred) / (_rms(x_trap) + 1e-6)
        factor = SAFETY * (tol / max(err, 1e-12)) ** 0.5
        if err <= tol or abs(dt_trial) <= MIN_DT:
            x = x_trap
            if sde_strength > 0.0:
                # Error-gated shake: gamma follows how hard this
                # step was (err/tol), capped by sde_strength. The
                # shake swaps part of the state's implied noise
                # for fresh noise along the RF interpolation line
                # x = (1-s)*x0 + s*eps, which keeps the state on
                # the flow manifold (blending the raw residual
                # (x - x0)/s as if it were unit noise does not -
                # an early draft of this formula diverged in the
                # mirror test and was replaced).
                gamma = min(err / tol, 1.0) * float(sde_strength)
                if gamma > 0.0:
                    d = x_trap - s_next * v_pred
                    gen = torch.Generator(device=x.device)
                    gen.manual_seed(
                        (int(noise_seed) + stats["accepted"])
                        % (1 << 63))
                    eps = torch.randn(x.shape, generator=gen,
                                      device=x.device, dtype=x.dtype)
                    keep = (1.0 - gamma * gamma) ** 0.5
                    eps_implied = (x_trap - (1.0 - s_next) * d) / s_next
                    x = ((1.0 - s_next) * d
                         + s_next * (keep * eps_implied + gamma * eps))
                    stats["shake_sum"] = (stats.get("shake_sum", 0.0)
                                          + gamma)
            if color_strength > 0.0 and 0.0 < s_next <= color_start:
                # Chroma steering: push this step's denoised
                # estimate away from its per-channel spatial mean
                # (colourfulness / feature contrast), ramping in
                # as sigma falls, and shift the accepted state by
                # the on-trajectory equivalent (1-s)*delta. The
                # shift is capped at 10% of the state's RMS per
                # step so the steering can bend the trajectory
                # but never run away with it (an uncapped draft
                # diverged in the mirror test).
                d = x_trap - s_next * v_pred
                dims = tuple(range(2, x.ndim))
                mu = d.mean(dim=dims, keepdim=True)
                w = 1.0 - s_next / float(color_start)
                gain = 1.0 + float(color_strength) * w
                d2 = mu + (d - mu) * gain
                delta = (1.0 - s_next) * (d2 - d)
                dr = _rms(delta)
                cap = 0.10 * _rms(x)
                if dr > cap and dr > 0.0:
                    delta = delta * (cap / dr)
                x = x + delta
            v = v_pred
            m = b * m + (1.0 - b) * v_pred
            t = t_next
            sigma = s_next
            stats["accepted"] += 1
            if callback is not None:
                callback({"x": x, "i": stats["accepted"] - 1,
                          "sigma": float(sigma),
                          "denoised": denoised_p})
            dt = dt_trial * min(max(factor, 0.2), GROW_MAX)
        else:
            dt = dt_trial * min(max(factor, SHRINK_MIN), 0.95)

    if t > 0.0:
        # Trial budget exhausted: finish with plain Euler steps.
        print(f"[SerenSampler] trial budget exhausted at t={t:.4f}; "
              "finishing with fixed Euler steps")
        remaining = max(1, int(round(t * 30)))
        grid_t = [t * (1.0 - k / remaining) for k in range(remaining + 1)]
        for k in range(remaining):
            s_cur = shift_sigma(grid_t[k], shift)
            s_nxt = (shift_sigma(grid_t[k + 1], shift)
                     if grid_t[k + 1] > 0.0 else 0.0)
            denoised = _denoised(model, x, s_cur, extra_args, stats)
            if s_nxt == 0.0:
                x = denoised
            else:
                vv = (x - denoised) / s_cur
                x = x + vv * (s_nxt - s_cur)
            stats["accepted"] += 1
            if callback is not None:
                callback({"x": x, "i": stats["accepted"] - 1,
                          "sigma": float(s_nxt), "denoised": denoised})
    return x


class SerenSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0,
                                  "max": 30.0, "step": 0.1}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                         "max": 10.0, "step": 0.1}),
                "tolerance": ("FLOAT", {"default": DEFAULT_TOLERANCE,
                                        "min": 0.0005, "max": 0.2,
                                        "step": 0.0005}),
                "beta": ("FLOAT", {"default": DEFAULT_BETA, "min": 0.0,
                                   "max": 0.95, "step": 0.05}),
                "max_steps": ("INT", {"default": 120, "min": 4,
                                      "max": 500}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "steps_used", "evals_used")
    FUNCTION = "sample"
    CATEGORY = "sampling/seren"

    def sample(self, model, positive, negative, latent_image, seed,
               cfg, flow_shift, tolerance, beta, max_steps):
        latent_samples, expanded = _fix_latent(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if expanded:
            noise = noise.unsqueeze(2)
        sigmas = flow_linear_shift_sigmas(max_steps, flow_shift)
        stats = {"accepted": 0, "trials": 0, "evals": 0}

        def sampler_function(model_, x, sigs, extra_args=None,
                             callback=None, disable=None, **kwargs):
            return _seren_loop(model_, x, extra_args or {}, callback,
                               tolerance, beta, flow_shift, max_steps,
                               stats)

        sampler = comfy.samplers.KSAMPLER(sampler_function)
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        print(f"[SerenSampler] tol={tolerance}, beta={beta}, "
              f"shift={flow_shift}, cfg={cfg} -> "
              f"accepted={stats['accepted']}, "
              f"trials={stats['trials']}, evals={stats['evals']}")
        out = dict(latent_image)
        if expanded:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out, stats["accepted"], stats["evals"])


class SerenSDESampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0,
                                  "max": 30.0, "step": 0.1}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                         "max": 10.0, "step": 0.1}),
                "tolerance": ("FLOAT", {"default": DEFAULT_TOLERANCE,
                                        "min": 0.0005, "max": 0.2,
                                        "step": 0.0005}),
                "beta": ("FLOAT", {"default": DEFAULT_BETA, "min": 0.0,
                                   "max": 0.95, "step": 0.05}),
                "max_steps": ("INT", {"default": 120, "min": 4,
                                      "max": 500}),
                "sde_strength": ("FLOAT", {"default": 0.3, "min": 0.0,
                                           "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "steps_used", "evals_used")
    FUNCTION = "sample"
    CATEGORY = "sampling/seren"

    def sample(self, model, positive, negative, latent_image, seed,
               cfg, flow_shift, tolerance, beta, max_steps,
               sde_strength):
        latent_samples, expanded = _fix_latent(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if expanded:
            noise = noise.unsqueeze(2)
        sigmas = flow_linear_shift_sigmas(max_steps, flow_shift)
        stats = {"accepted": 0, "trials": 0, "evals": 0}

        def sampler_function(model_, x, sigs, extra_args=None,
                             callback=None, disable=None, **kwargs):
            return _seren_loop(model_, x, extra_args or {}, callback,
                               tolerance, beta, flow_shift, max_steps,
                               stats, sde_strength=sde_strength,
                               noise_seed=seed)

        sampler = comfy.samplers.KSAMPLER(sampler_function)
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        mean_shake = (stats.get("shake_sum", 0.0)
                      / max(stats["accepted"], 1))
        print(f"[SerenSDE] tol={tolerance}, beta={beta}, "
              f"sde_strength={sde_strength}, shift={flow_shift}, "
              f"cfg={cfg} -> accepted={stats['accepted']}, "
              f"trials={stats['trials']}, evals={stats['evals']}, "
              f"mean_shake={mean_shake:.3f}")
        out = dict(latent_image)
        if expanded:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out, stats["accepted"], stats["evals"])


class SerenColorSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0,
                                  "max": 30.0, "step": 0.1}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                         "max": 10.0, "step": 0.1}),
                "tolerance": ("FLOAT", {"default": DEFAULT_TOLERANCE,
                                        "min": 0.0005, "max": 0.2,
                                        "step": 0.0005}),
                "beta": ("FLOAT", {"default": DEFAULT_BETA, "min": 0.0,
                                   "max": 0.95, "step": 0.05}),
                "max_steps": ("INT", {"default": 120, "min": 4,
                                      "max": 500}),
                "color_strength": ("FLOAT", {"default": 0.15,
                                             "min": 0.0, "max": 2.0,
                                             "step": 0.05}),
                "color_start": ("FLOAT", {"default": 0.35, "min": 0.1,
                                          "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "steps_used", "evals_used")
    FUNCTION = "sample"
    CATEGORY = "sampling/seren"

    def sample(self, model, positive, negative, latent_image, seed,
               cfg, flow_shift, tolerance, beta, max_steps,
               color_strength, color_start):
        latent_samples, expanded = _fix_latent(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if expanded:
            noise = noise.unsqueeze(2)
        sigmas = flow_linear_shift_sigmas(max_steps, flow_shift)
        stats = {"accepted": 0, "trials": 0, "evals": 0}

        def sampler_function(model_, x, sigs, extra_args=None,
                             callback=None, disable=None, **kwargs):
            return _seren_loop(model_, x, extra_args or {}, callback,
                               tolerance, beta, flow_shift, max_steps,
                               stats, color_strength=color_strength,
                               color_start=color_start)

        sampler = comfy.samplers.KSAMPLER(sampler_function)
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        print(f"[SerenColor] tol={tolerance}, beta={beta}, "
              f"color_strength={color_strength}, "
              f"color_start={color_start}, shift={flow_shift}, "
              f"cfg={cfg} -> accepted={stats['accepted']}, "
              f"trials={stats['trials']}, evals={stats['evals']}")
        out = dict(latent_image)
        if expanded:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out, stats["accepted"], stats["evals"])


# ---------------------------------------------------------- KSampler
# dropdown registration with fixed defaults (tolerance 0.01,
# beta 0.5, sde_strength 0.3 / color_strength 0.15 for the
# siblings, shift estimated from the incoming grid). The
# standalone nodes above remain the way to tune the knobs.
def _seren_registered(model, x, sigmas, extra_args=None,
                      callback=None, disable=None, **kwargs):
    shift = estimate_shift(sigmas)
    s0 = float(sigmas[0])
    t_start = shift_t(s0, shift) if s0 < 1.0 else 1.0
    stats = {"accepted": 0, "trials": 0, "evals": 0}
    out = _seren_loop(model, x, extra_args or {}, callback,
                      DEFAULT_TOLERANCE, DEFAULT_BETA, shift, 120,
                      stats, t_start=t_start)
    print(f"[SerenSampler] registered run: shift~{shift:.2f}, "
          f"accepted={stats['accepted']}, evals={stats['evals']}")
    return out


def _seren_sde_registered(model, x, sigmas, extra_args=None,
                          callback=None, disable=None, **kwargs):
    shift = estimate_shift(sigmas)
    s0 = float(sigmas[0])
    t_start = shift_t(s0, shift) if s0 < 1.0 else 1.0
    seed = int((extra_args or {}).get("seed", 0))
    stats = {"accepted": 0, "trials": 0, "evals": 0}
    out = _seren_loop(model, x, extra_args or {}, callback,
                      DEFAULT_TOLERANCE, DEFAULT_BETA, shift, 120,
                      stats, t_start=t_start, sde_strength=0.3,
                      noise_seed=seed)
    print(f"[SerenSDE] registered run: shift~{shift:.2f}, "
          f"accepted={stats['accepted']}, evals={stats['evals']}")
    return out


def _seren_color_registered(model, x, sigmas, extra_args=None,
                            callback=None, disable=None, **kwargs):
    shift = estimate_shift(sigmas)
    s0 = float(sigmas[0])
    t_start = shift_t(s0, shift) if s0 < 1.0 else 1.0
    stats = {"accepted": 0, "trials": 0, "evals": 0}
    out = _seren_loop(model, x, extra_args or {}, callback,
                      DEFAULT_TOLERANCE, DEFAULT_BETA, shift, 120,
                      stats, t_start=t_start, color_strength=0.15,
                      color_start=0.35)
    print(f"[SerenColor] registered run: shift~{shift:.2f}, "
          f"accepted={stats['accepted']}, evals={stats['evals']}")
    return out


def register_samplers():
    try:
        import comfy.samplers
        from comfy.k_diffusion import sampling as _k_sampling
        entries = {"seren": _seren_registered,
                   "seren_sde": _seren_sde_registered,
                   "seren_color": _seren_color_registered}
        for name, fn in entries.items():
            if name not in comfy.samplers.KSampler.SAMPLERS:
                comfy.samplers.KSampler.SAMPLERS.append(name)
            setattr(_k_sampling, f"sample_{name}", fn)
        print("[SerenSampler] registered KSampler samplers: "
              "seren, seren_sde, seren_color")
    except Exception as e:  # never break node loading
        print(f"[SerenSampler] sampler registration skipped: {e}")


register_samplers()


NODE_CLASS_MAPPINGS = {
    "SerenSampler": SerenSampler,
    "SerenSDESampler": SerenSDESampler,
    "SerenColorSampler": SerenColorSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SerenSampler": "Seren Sampler (momentum x adaptive)",
    "SerenSDESampler": "Seren SDE Sampler (error-shaken)",
    "SerenColorSampler": "Seren Color Sampler (chroma steering)",
}
