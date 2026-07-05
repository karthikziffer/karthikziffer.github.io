---
layout: post
title: "Synthetic Textile Data with Blender and Generative AI"
author: "Karthik"
categories: journal
tags: [agentic ai, prompt engineering]
---

This blog shares the latest version of our synthetic-data pipeline for generating better training data with Blender. We render a synthetic view of the textile on a table from different camera angles, then pass each render through a generative model to produce a photorealistic image, with variance such as crumpled or folded textures introduced through the prompt. This adds disturbances that Blender's simulation alone can't produce. A vision LLM judge then compares the generated image against the original synthetic render to verify that orientation and other similarity conditions still match.

This solves a real problem: without a reference image, camera-angle variance has to be described entirely through the prompt, which is imprecise. A synthetic render gives the model something concrete to match, so the angle stays under control. This synthetic data is used for pre-training, with a much smaller sample of real data reserved for fine-tuning.

## Setting the Blender scene: core concepts {#scene-setup}

### Asset sourcing {#asset-sourcing}

The T-shirt mesh is a downloaded 3D asset (glTF format, from Sketchfab) and the table is a separate downloaded asset (FBX format), rather than being modeled from scratch. glTF was preferred for the T-shirt since it's a lightweight, web-compatible format that keeps texture references bundled.

The table asset imported into the scene.

<img src="/assets/images/blender-synthetic-dataset/table-asset.png" alt="Table asset" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

The T-shirt asset imported into the scene.

<img src="/assets/images/blender-synthetic-dataset/tshirt-asset.png" alt="T-shirt asset" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

### Scene composition {#scene-composition}

The table is placed first, with the T-shirt on top. A small amount of overlap between the two is fine since the training data only needs "textile placed on a table," not pixel-perfect contact. The scene itself stays static; all the variation needed for training comes from randomizing the T-shirt's orientation and the camera's viewpoint, not from animating scene properties.





The full Blender setup, showing the table and T-shirt meshes, the cloth physics settings, and the camera in the outliner.

<img src="/assets/images/blender-synthetic-dataset/scene-setup-overview.png" alt="Blender scene setup overview" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />



### Camera perspective rationale {#camera-rationale}

The camera setup mirrors real-world deployment, where the recognition system captures the textile from a top-down angle, either a fixed overhead camera or one mounted on a robotic arm. To match those conditions, the synthetic camera angles are also sampled around a top-down viewpoint.



A sample render from the camera's own viewpoint, showing the T-shirt on the table against the HDRI warehouse backdrop.

<img src="/assets/images/blender-synthetic-dataset/camera-viewpoint-sample.png" alt="Sample camera placement viewpoint" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />


### Material representation {#material-representation}

A freshly imported 3D model has no shading by default; it renders as flat gray until a material is authored. Materials in Blender are built as a PBR node graph feeding a Principled BSDF shader. Three maps matter for textile specifically:
- Base Color: the surface color, in sRGB space.
- Roughness: matte vs. glossy, in Non-Color space.
- Normal map: fine surface detail like weave and folds, also Non-Color, converted into a usable normal vector before it reaches the shader.

Getting the color space right per map is the concept that matters: mixing it up (e.g., leaving a roughness map in sRGB) silently produces incorrect shading.

The T-shirt mesh with no material applied yet, rendered as flat gray.

<img src="/assets/images/blender-synthetic-dataset/no-shading.png" alt="No shading applied" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

The table's base color (albedo) texture used for its material.

<img src="/assets/images/blender-synthetic-dataset/table-basecolor.jpg" alt="Table base color texture" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

The table rendered with its material and shading applied.

<img src="/assets/images/blender-synthetic-dataset/with-shading.png" alt="With shading applied" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

Setting up the shading nodes for the table's material.

<img src="/assets/images/blender-synthetic-dataset/table-shading-setup-1.png" alt="Setting shading for the table, step 1" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

<img src="/assets/images/blender-synthetic-dataset/table-shading-setup-2.png" alt="Setting shading for the table, step 2" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

### Lighting concept {#lighting-concept}

Lighting is treated as a variable, not a constant, since the recognition model needs to generalize across real-world conditions like harsh factory light, dim shadows, and overcast daylight. Instead of hand-placing lights, we use an HDRI, a 360° photo of a real environment, as world lighting; it supplies both ambient light and realistic reflections in one step. We use free warehouse HDRIs (e.g., from Poly Haven) to match the target deployment setting: a textile sorting facility.

The HDRI environment map providing the scene's lighting and reflections.

<img src="/assets/images/blender-synthetic-dataset/hdri-lighting.png" alt="Light in HDRI" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

### Multi-view capture strategy {#multi-view-capture}

Instead of a single fixed shot, each iteration randomizes the T-shirt's rotation and orbits the camera within a positional range above the table, generating many viewpoints per simulated drape.

A Track-To constraint keeps the camera aimed at the table's center no matter where it's randomly placed, so every render stays framed on the subject instead of drifting into empty background. This also fixes the earlier bug where a bad camera placement clipped the textile out of frame.

The first sampled camera angle looking down at the table.

<img src="/assets/images/blender-synthetic-dataset/camera-angle-1.png" alt="Camera angle 1" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

The second sampled camera angle looking down at the table.

<img src="/assets/images/blender-synthetic-dataset/camera-angle-2.png" alt="Camera angle 2" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 32px auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />

### Rendering as the final step {#rendering-final-step}

Once mesh, material, lighting, and camera are set for a given iteration, rendering converts that 3D scene state into a single 2D image, which becomes one training sample. Repeating this loop with fresh randomization each time produces the dataset of varied synthetic renders that later get passed to the generative model.

Different synthetic renders of the T-shirt on the table, captured from varying camera angles.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; max-width: 700px; margin: 32px auto;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-1.png" alt="Synthetic render, camera angle A" style="display: block; width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/rendered-view-2.png" alt="Synthetic render, camera angle B" style="display: block; width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/rendered-view-3.png" alt="Synthetic render, camera angle C" style="display: block; width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/rendered-view-4.png" alt="Synthetic render, camera angle D" style="display: block; width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</div>

## Generating photorealistic images {#generating-images}

Each synthetic render is then passed through FLUX.2, Qwen to produce a photorealistic generated image.

<table style="width: 100%; max-width: 760px; margin: 32px auto; border-collapse: collapse;">
<tr>
<th style="text-align: center; padding: 8px; border-bottom: 1px solid rgba(0,0,0,0.15);">Synthetic render</th>
<th style="text-align: center; padding: 8px; border-bottom: 1px solid rgba(0,0,0,0.15);">FLUX.2 generated</th>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-1.png" alt="Synthetic rendered view 1" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-1-1.png" alt="FLUX.2 generated image, rendered view 1, variant 1" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/rendered-view-1-flux2-klein-4b_transform-the-synthetic-rendered.png" alt="FLUX.2 generated image, rendered view 1, variant 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-2.png" alt="Synthetic rendered view 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-2-2.png" alt="FLUX.2 generated image, rendered view 2, variant 1" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/rendered-view-2-flux2-dev_transform-the-synthetic-rendered-image.png" alt="FLUX.2 generated image, rendered view 2, variant 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-3.png" alt="Synthetic rendered view 3" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-3-flux2-klein-4b_transform-the-synthetic-rendered%20(1).png" alt="FLUX.2 generated image, rendered view 3" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-4.png" alt="Synthetic rendered view 4" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-4.webp" alt="FLUX.2 generated image, rendered view 4, variant 1" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/rendered-view-4-image%20(1).webp" alt="FLUX.2 generated image, rendered view 4, variant 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
</table>

<table style="width: 100%; max-width: 760px; margin: 32px auto; border-collapse: collapse;">
<tr>
<th style="text-align: center; padding: 8px; border-bottom: 1px solid rgba(0,0,0,0.15);">Synthetic render</th>
<th style="text-align: center; padding: 8px; border-bottom: 1px solid rgba(0,0,0,0.15);">Qwen generated</th>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/Untitled.png" alt="Synthetic rendered view, untitled" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/Untitled-qwen-image-edit-2511_heres-a-cleaner-more.png" alt="Qwen generated image, variant 1" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/Untitled-qwen-image-edit-2511_heres-a-revised-version-of.png" alt="Qwen generated image, variant 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/Untitled-qwen-image-edit-2511_heres-a-stronger-more.png" alt="Qwen generated image, variant 3" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/Untitled-qwen-image-edit-2511_transform-the-synthetic.png" alt="Qwen generated image, variant 4" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/Untitled-qwen-image-edit-2511_use-the-t-shirt-placement.png" alt="Qwen generated image, variant 5" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-3.png" alt="Synthetic rendered view 3" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-3-qwen-image-edit-2511_use-the-modern-graphic-t.png" alt="Qwen generated image, rendered view 3, variant 1" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto 12px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/rendered-view-3-qwen-image-edit-2511_use-the-t-shirt-placement%20(1).png" alt="Qwen generated image, rendered view 3, variant 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
</table>

<table style="width: 100%; max-width: 760px; margin: 32px auto; border-collapse: collapse;">
<tr>
<th style="text-align: center; padding: 8px; border-bottom: 1px solid rgba(0,0,0,0.15);">Synthetic render</th>
<th style="text-align: center; padding: 8px; border-bottom: 1px solid rgba(0,0,0,0.15);">flux_edit generated</th>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-2.png" alt="Synthetic rendered view 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-2-flux_edit_1783241841.png" alt="flux_edit generated image, rendered view 2" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-3.png" alt="Synthetic rendered view 3" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-3-flux_edit_1783241778.png" alt="flux_edit generated image, rendered view 3" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
<tr>
<td style="padding: 8px; text-align: center;"><img src="/assets/images/blender-synthetic-dataset/rendered-view-4.png" alt="Synthetic rendered view 4" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" /></td>
<td style="padding: 8px; text-align: center;">
<img src="/assets/images/blender-synthetic-dataset/rendered-view-4-flux_edit_1783241885.png" alt="flux_edit generated image, rendered view 4" style="display: block; width: 100%; max-width: 320px; height: auto; margin: 0 auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</td>
</tr>
</table>

Both the distilled and base variants of the model were tried, and they come with a clear time-complexity trade-off: the distilled variant (4 steps) takes 10–20s per image, while the base variant (50 steps) takes approximately 2 minutes, even on an optimized inference engine.

## Failed generations {#failed-generations}

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; max-width: 700px; margin: 32px auto;">
<img src="/assets/images/blender-synthetic-dataset/failed1.png" alt="Collapsed table" style="display: block; width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
<img src="/assets/images/blender-synthetic-dataset/failed_flux_edit_1783242041.png" alt="Enlightened T-shirt" style="display: block; width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.12);" />
</div>

- **Collapsed table**: the table is collapsed, leaving the T-shirt floating with almost no surface context.
- **Enlightened T-shirt**: a ghosted, duplicated T-shirt artifact in the upper-left of the generated image.


## Evaluation {#evaluation}

The simplest evaluation uses a vision LLM judge: it compares the synthetic render against the generated image and returns a similarity score, covering orientation, T-shirt placement, and relative scale.

### Judging modes we use {#judging-modes}

| Mode | Image editing | T2I generation |
|---|---|---|
| Prompt tuning | Edit + evaluate, then refine the prompt | Generate + evaluate, then refine the prompt |
| Pointwise | Score one edit against its source + instruction | Score one generation against its prompt |
| Pairwise | Compare two edits, pick the better one | Compare two generations, pick the better one |



### Evaluation metrics {#evaluation-metrics}

- **How many of the images match the similarity** – "Similarity" here is semantic, not pixel-level: the FLUX.2 pass is expected to change texture, color, and photorealistic style, since that's the whole point of the transformation. What must be preserved:

| Constraint | What it checks |
|---|---|
| Orientation | The T-shirt's facing matches the synthetic reference. |
| Placement | The T-shirt stays fully on the table surface, with no floating or clipping outside the table bounds. |
| Scale | The table reads as proportionately larger than the T-shirt, not visually shrunk or dominating. |
| Integrity | Exactly one T-shirt, no duplicated or ghosted garments. |
| Pose | The fold state is still recognizable as the same pose as the synthetic render. |

  A generated image "matches" only if the vision LLM judge scores it as satisfying all of these constraints, not just some of them.

- **What are the improvements that could be done** – This splits into two levels:
  - **Image generation quality** – better prompt conditioning to preserve scene geometry (table boundaries, occlusion regions), and choosing the base (50-step) model over the distilled (4-step) one when fidelity matters more than generation speed.
  - **Pipeline level** – replacing a purely vision-LLM-judge similarity score with the physics-informed metric noted below, and feeding failure cases back into prompt iteration rather than treating each generation as a one-shot attempt.


In practice, we score each generated image using a weighted combination of the vision LLM judge and classical image-processing checks, and use that combined score to validate the image before it's admitted into the pre-training pipeline.




### Insights from our data {#insights}

1. Occlusion regions are where the model struggles most: samples with occluded T-shirt or table regions consistently showed generation issues.
2. A physics-informed metric would catch these failure modes more reliably than the current vision LLM judge alone.
3. Getting to the best generated image took several rounds of prompt iteration rather than a single pass.

## Conclusion {#conclusion}

Pairing Blender's synthetic renders with FLUX.2/Qwen/flux_edit generation gives us a controllable way to inject real-world variance such as lighting and disturbances that the simulation alone can't produce, while the vision LLM judge keeps generated images anchored to the original orientation, placement, and scale. The main open gap is occlusion handling, which is where we're focusing next, alongside folding in a physics-informed metric to complement the LLM judge.
