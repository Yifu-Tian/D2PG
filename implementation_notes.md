# D^2 PG Implementation Notes

## Suggested Repository Shape

```text
d2pg/
  README.md
  research_plan.md
  implementation_notes.md
  configs/
  data/
  src/
  experiments/
```

The current folder is a research scratchpad. Code can be added once the target dataset is chosen.

## Minimal Data Record

Each training item should eventually contain:

```text
{
  "image_t": ...,
  "image_t1": ...,
  "degraded_image_t": ...,
  "degraded_image_t1": ...,
  "T_t_t1": ...,
  "pose_r9": ...,
  "degradation": {
    "brightness": ...,
    "blur": ...,
    "occlusion": ...
  }
}
```

## Pose Vector Convention

Use:

```text
pose_r9 = [tx, ty, tz, r6_1, r6_2, r6_3, r6_4, r6_5, r6_6]
```

The 6D rotation representation can be decoded by:

1. Treat first 3 values as vector `a1`.
2. Treat next 3 values as vector `a2`.
3. Normalize `a1` into basis vector `b1`.
4. Remove from `a2` the component parallel to `b1`, then normalize into `b2`.
5. Compute `b3 = cross(b1, b2)`.
6. Rotation matrix is `[b1, b2, b3]`.

This keeps the generated rotation valid.

## First Model Baseline

Start with direct regression:

```text
condition = encoder(degraded_image_t, degraded_image_t1)
pose_r9 = MLP(condition)
```

This baseline is important. Without it, it will be hard to prove diffusion helped.

## First Diffusion Model

Use a small conditional MLP:

```text
input: noisy_pose_r9, time_embedding, image_condition
output: predicted_noise or predicted_clean_pose
```

Recommended first target:

```text
epsilon prediction
```

because it follows the standard diffusion training recipe.

## First Flow-Matching Model

Use linear interpolation:

```text
y_tau = (1 - tau) * y_0 + tau * y_1
```

where:

```text
y_0 = Gaussian noise or previous-pose prior
y_1 = ground-truth pose_r9
```

Train:

```text
v_theta(y_tau, tau, c) ~= y_1 - y_0
```

This is simple and close to the flow-matching notes already in `track.md`.

## Practical Evaluation Order

1. Overfit 100 samples.
2. Train direct regression on clean image pairs.
3. Train direct regression on degraded image pairs.
4. Train diffusion on degraded image pairs.
5. Train flow matching on degraded image pairs.
6. Add previous-pose initialization.
7. Roll predictions into short trajectories.

## Notes For Writing

Possible paper-style title:

```text
D^2 PG: Diffusion Degraded Pose Generator for Robust Visual Odometry
```

Possible claim:

```text
Visual odometry under degradation is inherently ambiguous; conditional generative pose estimation can produce calibrated pose hypotheses instead of brittle single-point estimates.
```

Avoid overclaiming:

```text
Do not claim diffusion replaces geometry.
```

Better framing:

```text
Diffusion provides robust pose hypotheses that can complement geometric back-ends.
```

