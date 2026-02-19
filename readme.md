# DeGhost



Panoramic microscopy images generated using the Olympus Stream software may exhibit stitching artifacts when adjacent tiles are imperfectly aligned. These artifacts, commonly referred to as ghost artifacts, occur when overlapping regions of neighboring images are slightly mismatched and blended together. As a result, structural features—such as fiber edges in composite micrographs—appear duplicated, blurred, or spatially shifted.

This effect can be described mathematically as a linear superposition of the original image and a shifted version of itself:

$$I_{\text{ghost}} = (1 - \alpha) \cdot I + \alpha \cdot \text{shift}(I),$$

where:
- $I$ denotes the clean image,
- $\text{shift}(I)$ represents a spatially shifted copy,
- $\alpha \in [0,1]$ controls the blending strength.

In this formulation, ghosting is not a blur in the classical sense, but rather a structured misalignment artifact resulting from image blending.

---

Residual Learning for Deghosting

Reconstructing the clean image directly from $I_{\text{ghost}}$ is possible but suboptimal from a learning perspective. Instead, we reformulate the problem in terms of residual prediction.

From the forward model, the clean image can be written as:

$$I = I_{\text{ghost}} + r,$$

where the residual is

$$r = I - I_{\text{ghost}}.$$

Substituting the ghost model yields:

$$r = \alpha \left(I - \text{shift}(I)\right).$$

This shows that the residual is a structured, spatially correlated signal that encodes the difference between aligned and misaligned content.

Learning this residual directly offers several advantages:
- Reduced output magnitude: The residual is typically much smaller than the image intensity range, making optimization more stable.
- Identity-safe behavior: If no artifact is present, the optimal residual is zero.
- Faster convergence: The network focuses on modeling only the artifact component rather than reconstructing the full image.
- Sharper reconstruction: Since the underlying image is already sharp, subtracting the structured residual restores edges without requiring explicit gradient constraints.

Therefore, the DeGhost-UNet architecture is designed to predict the residual r, and the final reconstruction is obtained via:

$$\hat{I}_{\text{clean}} = I_{\text{ghost}} + r_\theta,$$

where $r_\theta$ is the network prediction.

This residual formulation aligns naturally with the physics of stitching artifacts and results in improved stability, faster convergence, and better preservation of fine fiber structures compared to direct image-to-image mapping.

## Model

We exploit a simple UNet-Style architecture for 


## Loss functions

$$\mathcal{L}_{total} = \mathcal{L}_{\text{res}} + \lambda_{\text{edge}} \cdot \mathcal{L}_{\text{edge}}$$


### Residual Loss

- MSE (not recommended)

$$\mathcal{L}_{\text{res}} = \frac{1}{N} \sum_{i=1}^{N} \|r_\theta^{(i)} - r^{(i)}\|_2^2$$

- MAE (recommended)

$$\mathcal{L}_{\text{res}} = \frac{1}{N} \sum_{i=1}^{N} \|r_\theta^{(i)} - r^{(i)}\|_1$$

- Charbonnier Loss (recommended)

$$\mathcal{L}_{\text{res}} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{\|r_\theta^{(i)} - r^{(i)}\|_2^2 + \epsilon^2}$$

### Edge Loss

To further enhance the preservation of fine fiber structures, we introduce an edge-aware loss component. This loss encourages the network to produce residual predictions that not only minimize pixel-wise differences but also maintain sharp edges.

$$\mathcal{L}_{\text{edge}} = \frac{1}{N} \sum_{i=1}^{N} \| \nabla r_\theta^{(i)} - \nabla r^{(i)} \|_1$$

where $\nabla$ denotes a spatial gradient operator (e.g., Laplacian filter) applied to the residuals. By penalizing discrepancies in edge information, this loss helps the model focus on accurately reconstructing fiber boundaries and reduces blurring artifacts in the final output.

### SSIM Loss

To further improve the perceptual quality of the deghosted images, we can incorporate a Structural Similarity Index Measure (SSIM) loss component. This loss encourages the network to produce outputs that are not only pixel-wise accurate but also structurally similar to the ground truth clean images.

$$\mathcal{L}_{\text{SSIM}}=1-\text{SSIM}(I_{\text{clean}}, \hat{I}_{\text{clean}})$$

where $\text{SSIM}$ is computed between the ground truth clean image $I_{\text{clean}}$ and the reconstructed image $\hat{I}_{\text{clean}}$. By minimizing this loss, the model is incentivized to preserve important structural features and textures in the deghosted output, leading to visually more appealing results.

$$\text{SSIM}(x,y) =
\frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where $\mu_x$ and $\mu_y$ are the mean intensities, $\sigma_x^2$ and $\sigma_y^2$ are the variances, and $\sigma_{xy}$ is the covariance between the two images. $C_1=(k_1\,L)^2$ and $C_2=(k_2\,L)^2$ are small constants to stabilize the division.

SSIM compares:
- Luminance similarity
- Contrast similarity
- Structural similarity

