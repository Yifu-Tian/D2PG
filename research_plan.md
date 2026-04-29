# D^2 PG Research Plan

## 1. Problem Definition

We want to estimate relative camera motion under degraded visual conditions.

Given two consecutive observations:

```text
I_t, I_{t+1}
```

and optional context:

```text
previous poses, IMU, optical flow, degradation level
```

predict the relative transform:

```text
T_{t -> t+1} in SE(3)
```

D^2 PG models this transform through a Euclidean vector:

```text
y in R^9
y = [t_x, t_y, t_z, r_1, r_2, r_3, r_4, r_5, r_6]
```

where the last 6 numbers are decoded into a valid rotation matrix.

## 2. Why Deterministic VO Struggles

Classic VO and SLAM front-ends usually depend on stable visual evidence:

- Detect corners or features.
- Match them across frames.
- Estimate relative motion from geometry.
- Reject outliers.

This works beautifully when the world is well lit and textured. It becomes fragile when:

- The image is too dark.
- Motion blur smears features.
- The camera sees a plain wall or floor.
- Dynamic objects occlude static landmarks.
- Exposure changes destroy photometric consistency.

In those cases, the image pair may support multiple plausible camera motions. A single deterministic output can be misleading.

## 3. Generative Reformulation

Instead of:

```text
y_hat = f_theta(I_t, I_{t+1})
```

we train a conditional generative model:

```text
y ~ p_theta(y | c)
```

where `c` is a condition extracted from degraded observations.

Possible conditions:

- CNN / ViT features from the image pair.
- RAFT-style optical-flow features.
- Previous relative pose sequence.
- IMU preintegration.
- Degradation embedding, such as blur level or brightness level.

The simplest version uses only the degraded image pair.

## 4. Diffusion Formulation

Let the clean target pose vector be:

```text
y_0 in R^9
```

The forward noising process creates:

```text
y_tau = alpha_tau y_0 + sigma_tau epsilon
epsilon ~ N(0, I)
```

The model learns to denoise:

```text
epsilon_theta(y_tau, tau, c)
```

or directly predict the clean pose:

```text
y_theta(y_tau, tau, c)
```

At inference time:

```text
noise / prior pose -> denoising steps -> relative pose sample
```

If multiple samples are generated, the system can estimate uncertainty:

```text
mean pose = average of samples
uncertainty = sample covariance / dispersion
```

## 5. Flow-Matching Variant

A flow-matching version may be cleaner and closer to the A2A paper.

Noise-to-pose:

```text
z_0 ~ N(0, I)
z_1 = pose latent
```

Motion-informed pose-to-pose:

```text
z_0 = encoder(previous pose deltas or IMU prior)
z_1 = encoder(current relative pose)
```

The second version is more interesting because it mirrors the A2A insight:

> In a temporal system, the previous motion is a better starting point than random Gaussian noise.

## 6. Model Sketch

### Encoder

Inputs:

```text
I_t, I_{t+1}
```

Possible architecture:

- Shared CNN or ResNet backbone.
- Feature correlation or simple concatenation.
- MLP projection to condition vector `c`.

### Pose Generator

Inputs:

```text
noisy pose y_tau
time tau
condition c
```

Output:

```text
denoised pose / velocity / noise prediction
```

Possible architecture:

- MLP for the first prototype.
- Transformer if using pose history.
- Small UNet-style MLP blocks if diffusion steps are many.

### Pose Decoder

For `R^9`:

- First 3 dimensions become translation.
- Last 6 dimensions become rotation using 6D-to-SO(3) projection.
- Compose final `SE(3)` relative transform.

## 7. Data Strategy

Start with datasets that already provide ground-truth or high-quality poses:

- EuRoC MAV
- TUM RGB-D
- KITTI Odometry
- RealSense / Vicon data if available locally

Create degraded training inputs synthetically:

- Darkness: gamma correction, exposure reduction, shot noise.
- Motion blur: random linear or rotational kernels.
- Occlusion: random masks, dynamic object masks.
- Texture loss: blur + low contrast + compression.

Training sample:

```text
clean image pair -> apply degradation -> degraded image pair
ground-truth pose delta -> R^9 target
```

## 8. Baselines

Start with simple baselines before comparing to full SLAM systems:

- Direct regression: image pair -> `R^9` pose.
- Regression with uncertainty head.
- Diffusion pose generator from Gaussian noise.
- Flow matching from Gaussian noise.
- Motion-informed flow matching using previous pose deltas.

Later comparisons:

- ORB-SLAM style tracking failure rate.
- VINS / OpenVINS / DROID-SLAM if integration is practical.

## 9. Metrics

Single-step pose metrics:

- Translation error.
- Rotation error.
- Relative Pose Error (RPE).

Trajectory metrics:

- Absolute Trajectory Error (ATE).
- Tracking failure rate.
- Drift over time.

Generative metrics:

- Negative log-likelihood proxy.
- Calibration of uncertainty.
- Whether ground-truth pose lies inside high-probability samples.
- Diversity versus accuracy tradeoff.

## 10. Minimal Prototype

The first prototype should avoid full SLAM complexity.

Milestone 1:

```text
Train direct regression baseline on degraded image pairs.
```

Milestone 2:

```text
Train diffusion model to generate R^9 relative poses.
```

Milestone 3:

```text
Sample K candidate poses and compare mean / best-of-K / uncertainty.
```

Milestone 4:

```text
Add previous pose deltas as a motion prior.
```

Milestone 5:

```text
Roll relative poses into a trajectory and evaluate ATE / RPE.
```

## 11. Key Research Questions

1. Does diffusion help under severe degradation, or does it only add complexity?
2. Is `R^9 = translation + 6D rotation` stable enough for pose diffusion?
3. Does multi-sample generation provide meaningful uncertainty?
4. Does a motion-informed initialization outperform pure Gaussian noise?
5. Can this become a front-end proposal generator for a downstream SLAM back-end?

## 12. Risks

The model may learn dataset bias instead of geometry.

The generated pose distribution may be diverse but not physically consistent.

Synthetic degradation may not transfer to real darkness and blur.

Diffusion may be slower than needed unless step count is reduced.

Pose averaging is nontrivial because rotations live on `SO(3)`, even if generation happens in `R^9`.

## 13. Strongest Version Of The Idea

The strongest form of D^2 PG is not just:

```text
Use diffusion to regress camera pose.
```

It is:

```text
Use conditional generation to produce uncertainty-aware pose hypotheses when visual evidence is degraded, then let temporal consistency or a SLAM back-end select and refine them.
```

That framing makes the idea more defensible. It avoids claiming that diffusion magically solves geometry, and instead positions diffusion as a robust hypothesis generator under ambiguity.

