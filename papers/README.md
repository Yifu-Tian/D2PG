# D2PG Related Papers

This folder collects papers related to **D2PG: Diffusion Degraded Pose Generator**.

The papers are grouped by how they support the idea:

## Diffusion / Generative Camera Pose

### `2023_PoseDiffusion.pdf`

**PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment**

Why it matters:

- Closest conceptual neighbor to D2PG.
- Uses diffusion for camera pose estimation.
- Combines generative sampling with geometric bundle-adjustment style constraints.
- Useful for arguing that pose estimation can be formulated probabilistically instead of as direct regression.

Main difference from D2PG:

- Focuses on multi-view pose / SfM style estimation.
- Does not specifically target degraded VIO drift correction.

### `2023_ID-Pose.pdf`

**ID-Pose: Sparse-view Camera Pose Estimation by Inverting Diffusion Models**

Why it matters:

- Uses diffusion models to infer relative camera pose.
- Shows another path from generative image modeling to camera pose estimation.

Main difference from D2PG:

- More object-centric / sparse-view oriented.
- Not designed as a VIO correction module.

### `2025_BADGR.pdf`

**BADGR: Bundle Adjustment Diffusion Conditioned by Gradients**

Why it matters:

- Demonstrates diffusion combined with bundle-adjustment-style reasoning.
- Useful for designing D2PG as a geometry-aware generator rather than a pure black-box regressor.

Main difference from D2PG:

- Targets floor-plan / layout reconstruction, not degraded visual odometry.

## Probabilistic Pose Estimation

### `2022_RelPose.pdf`

**RelPose: Predicting Probabilistic Relative Rotation for Single Objects in the Wild**

Why it matters:

- Strong support for the claim that relative pose can be multi-modal.
- Shows why a single deterministic pose output can be insufficient.

Main difference from D2PG:

- Predicts object relative rotation, not camera trajectory or VIO pose correction.
- Not diffusion-based.

## Learned VO / SLAM Baselines

### `2021_DROID-SLAM.pdf`

**DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras**

Why it matters:

- Important learned SLAM baseline.
- Uses recurrent updates and dense bundle adjustment.
- A natural baseline or comparison point for learned pose refinement.

Main difference from D2PG:

- Deterministic iterative optimization rather than generative multi-hypothesis correction.

### `2018_DeepV2D.pdf`

**DeepV2D: Video to Depth with Differentiable Structure from Motion**

Why it matters:

- Classic learned depth-and-pose / visual odometry direction.
- Useful background for neural geometric estimation.

Main difference from D2PG:

- Not focused on diffusion or uncertainty-aware degraded VIO.

### `2022_DiffPoseNet.pdf`

**DiffPoseNet: Direct Differentiable Camera Pose Estimation**

Why it matters:

- Uses differentiable geometric constraints for camera pose estimation.
- Useful reminder that D2PG should preserve geometry and not become pure image-to-pose guessing.

Main difference from D2PG:

- `Diff` means differentiable, not diffusion.
- Does not generate multiple correction hypotheses.

## Drift Correction

### `2024_Probabilistic_Drift_Correction_VI_SLAM.pdf`

**A Probabilistic-based Drift Correction Module for Visual Inertial SLAMs**

Why it matters:

- Directly related to VIO / SLAM drift correction.
- Helpful for framing D2PG as a drift-aware correction module.

Main difference from D2PG:

- Not diffusion-based.
- Uses probabilistic correction with external prior information rather than generative pose hypotheses from degraded observations.

## Suggested Reading Order

1. `2023_PoseDiffusion.pdf`
2. `2022_RelPose.pdf`
3. `2024_Probabilistic_Drift_Correction_VI_SLAM.pdf`
4. `2021_DROID-SLAM.pdf`
5. `2025_BADGR.pdf`
6. `2023_ID-Pose.pdf`
7. `2022_DiffPoseNet.pdf`
8. `2018_DeepV2D.pdf`

## D2PG Positioning

D2PG should not be positioned as simply:

```text
diffusion for pose regression
```

A stronger positioning is:

```text
conditional generative pose correction for degraded VIO, producing multiple local correction hypotheses and uncertainty estimates for a downstream geometric backend.
```

