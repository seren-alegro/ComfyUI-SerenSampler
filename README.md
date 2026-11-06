# ComfyUI-SerenSampler

**Seren Sampler** - an experimental sampler for rectified flow
models (Anima family), for ComfyUI.

## The idea

Two earlier packs meet in one mechanism:

- From FlowMomentum: a **momentum** vector, an exponential moving
  average of past velocities.
- From AdaptiveFlow: **tolerance control** - no fixed step grid;
  each step must earn its size.

Seren uses the momentum as a *predictor*. It proposes the next
point for free (`x_pred = x + m * dSigma`); a single model
evaluation there plus a trapezoid correction gives the candidate
step, and the gap between prediction and correction is the error
estimate that grows or shrinks the next step. A typical accepted
step costs **one** model evaluation - the error estimate does not
need a second one.

This is a deterministic rectified-flow ODE walk: no stochastic
noise term, no look-ahead evaluation blended back into the step,
no adaptive noise. The momentum never touches the model input; it
only proposes points and measures surprise.

## Node

**Seren Sampler (momentum x adaptive)** - `sampling/seren`

Inputs:

- `seed`, `cfg`, `flow_shift` - as usual for this family of packs.
- `tolerance` (default 0.01) - allowed per-step prediction error,
  relative RMS. Smaller = slower and more faithful.
- `beta` (default 0.5) - momentum memory: 0 follows only the last
  velocity, higher values remember longer. Long memory predicts
  smoothly but can go stale where the trajectory curves hard.
- `max_steps` (default 120) - budget cap, not a step count. If the
  trial budget (2 x max_steps) ever runs out, the remainder is
  finished with plain fixed-step Euler.

Outputs: `latent`, `steps_used`, `evals_used` - so you can see
what a run actually cost.

**Seren SDE Sampler (error-shaken)** - `sampling/seren`

The stochastic sibling. After each accepted step, part of the
state's implied noise is swapped for fresh noise along the
rectified-flow interpolation line. The swap amount is gated by
how hard the step just was (`err / tolerance`, capped by
`sde_strength`, default 0.3): calm stretches stay nearly
deterministic, uncertain stretches get shaken. The noise follows
the local error topography of this walk - not a schedule tied to
overall progress - and the tolerance controller still governs
step sizes. Extra input: `sde_strength` (0-1; 0 = base Seren).

**Seren Color Sampler (chroma steering)** - `sampling/seren`

Colour handled inside the walk instead of as an after-the-fact
boost: in the low-sigma stretch (`sigma <= color_start`, default
0.35) where colour settles, each accepted step's denoised
estimate is pushed away from its per-channel spatial mean and
the state is shifted to match, bending the trajectory toward
richer colour while the model keeps re-denoising it. The
per-step shift is capped at 10% of the state RMS so the steering
cannot run away. Extra inputs: `color_strength` (default 0.35),
`color_start`.

## KSampler registration

The pack also registers **`seren`**, **`seren_sde`** and
**`seren_color`** in the standard KSampler `sampler_name`
dropdown, with fixed defaults (tolerance 0.01, beta 0.5;
sde_strength 0.3 / color_strength 0.35 for the siblings; shift
estimated from the incoming sigma grid). Use any scheduler with
them; the walk adapts its own steps between the grid's
endpoints. The standalone nodes remain the way to tune the
knobs.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/seren-alegro/ComfyUI-SerenSampler
```

Then restart ComfyUI.

## Example

`examples/seren_sampler_basic.json` - a minimal workflow using
the standalone Seren Sampler node (core nodes only; swap in your
own checkpoint, TE and VAE loaders as needed).

## Tests

`tests/test_seren.py` - numpy mirror of the core walk (no torch
needed): tolerance sweep, evaluation economy (~1 eval per step),
beta variations, and the trial-budget fallback.
`tests/test_seren_sde.py` - shake mirror: strength 0 is exactly
deterministic, strength > 0 diversifies endpoints and completes
in budget (error bounds are loose by design - the test field is
expansive and punishes any stochasticity far more than a trained
denoiser does). `tests/test_seren_color.py` - steering mirror:
strength 0 is exactly the base walk, strength > 0 stays finite
and moves the endpoint.

## License

MIT

---

## 한국어 요약

FlowMomentum의 모멘텀과 AdaptiveFlow의 허용 오차 제어를 하나로
합친 실험 샘플러예요. 모멘텀이 다음 지점을 공짜로 예측하고, 거기서
모델을 한 번만 평가해 보정하면, 예측과 보정의 차이가 곧 오차
추정이 돼요. 그래서 채택 스텝 하나에 모델 호출이 보통 한 번이면
되고, 스텝 크기는 tolerance가 알아서 조절해요. 결정적 rectified
flow 전용이며, KSampler 드롭다운에도 `seren`이라는 이름으로
등록돼요.

자매 샘플러가 둘 있어요. `seren_sde`는 확률형으로, 채택된 스텝이
얼마나 힘들었는지(오차/허용치)에 비례해 그 스텝의 노이즈 일부를
새 노이즈로 갈아 끼워요 — 잔잔한 구간은 거의 결정적이고, 불안한
구간만 흔들려요. `seren_color`는 색을 결과물에 사후 보정하지
않고 보행 도중에 다뤄요: 색이 자리 잡는 낮은 시그마 구간에서
매 스텝의 예측 완성본을 채널 평균에서 밀어내 궤적 자체를
풍부한 색 쪽으로 굽혀요.
