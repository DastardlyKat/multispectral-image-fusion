import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing import preprocess, vis_path, ir_path
from fusion import simpleAverageFusion, principleComponentAnalysis, discreteWaveletTransform

## Save fused images to a folder
def save_results(fusion_results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file, fused in fusion_results:
        fused_uint8 = (fused * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, file), fused_uint8)
    print(f"Saved {len(fusion_results)} images to {output_folder}")

## Visualize visible / infrared / fused images side by side
def visualize_results(preprocessed_pairs, avg_results, pca_results, dwt_results, n_samples=3):
    avg_lookup = {file: fused for file, fused in avg_results}
    pca_lookup = {file: fused for file, fused in pca_results}
    dwt_lookup = {file: fused for file, fused in dwt_results}

    for file, vis, ir in preprocessed_pairs[:n_samples]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Fusion Comparison - {file}", fontsize=14, fontweight='bold')

        # Row 1 — Source images + Average
        axes[0, 0].imshow(vis, cmap='gray')
        axes[0, 0].set_title("Visible", fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(ir, cmap='gray')
        axes[0, 1].set_title("Infrared (Thermal)", fontsize=12)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(avg_lookup[file], cmap='gray')
        axes[0, 2].set_title("Average Fusion", fontsize=12)
        axes[0, 2].axis('off')

        # Row 2 — PCA + Wavelet + empty slot
        axes[1, 0].imshow(pca_lookup[file], cmap='gray')
        axes[1, 0].set_title("PCA Fusion", fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(dwt_lookup[file], cmap='gray')
        axes[1, 1].set_title("Wavelet DWT Fusion", fontsize=12)
        axes[1, 1].axis('off')

        # Hide the empty 6th slot
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f"comparison_{file}.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved comparison_{file}.png")

if __name__ == "__main__":
    pairs = preprocess(vis_path, ir_path, limit=5)
    avg_results = simpleAverageFusion(pairs)
    pca_results = principleComponentAnalysis(pairs)
    dwt_results = discreteWaveletTransform(pairs)
    save_results(avg_results, "output/average")
    save_results(pca_results, "output/pca")
    save_results(dwt_results, "output/wavelet")
    visualize_results(pairs, avg_results, pca_results, dwt_results, n_samples=3)