# $D^2$ PG: Diffusion Degraded Pose Generator

**A Generative Approach to Visual Odometry in the Dark & Blur**

Traditional Visual Odometry (VO) and SLAM front-ends heavily rely on clear textures and stable lighting. When faced with severe motion blur, occlusion, or textureless walls, tracking easily gets lost. **DDP-Prior** aims to solve this by treating pose estimation in degraded environments as a *conditional generation problem*.

By leveraging a conditional diffusion model operating in Euclidean space ($\mathbb{R}^9$), this project generates robust relative pose hypotheses (3D Translation + Rotation 6D) given a clean previous frame and a degraded current frame.
