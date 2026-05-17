import numpy as np
from preprocessing import preprocess, vis_path, ir_path
from fusion import simpleAverageFusion, principleComponentAnalysis, discreteWaveletTransform
from metrics import compute_all_metrics, summarize_metrics
from utils import save_results, visualize_results

## Configuration
LIMIT = 20
N_SAMPLES = 3

## Step 1: Preprocessing
print("Loading and preprocessing images...")
pairs = preprocess(vis_path, ir_path, limit=LIMIT)
print(f"Loaded {len(pairs)} image pairs")

## Step 2: Fusion
print("\nRunning fusion methods...")
avg_results = simpleAverageFusion(pairs)
print("  Average fusion done")
pca_results = principleComponentAnalysis(pairs)
print("  PCA fusion done")
dwt_results = discreteWaveletTransform(pairs)
print("  Wavelet DWT fusion done")

## Step 3: Save fused images
print("\nSaving fused images...")
save_results(avg_results, "output/average")
save_results(pca_results, "output/pca")
save_results(dwt_results, "output/wavelet")

## Step 4: Compute metrics
print("\nComputing metrics...")
avg_metrics = compute_all_metrics(avg_results, pairs)
pca_metrics = compute_all_metrics(pca_results, pairs)
dwt_metrics = compute_all_metrics(dwt_results, pairs)

## Step 5: Print metric summaries
print("\n========== RESULTS ==========")
summarize_metrics(avg_metrics, "Simple Average Fusion")
summarize_metrics(pca_metrics, "PCA Fusion")
summarize_metrics(dwt_metrics, "Wavelet DWT Fusion")

## Step 6: Print comparison table
print("\n========== COMPARISON TABLE ==========")
keys = ['entropy', 'spatial_frequency', 'ssim_vis', 'ssim_ir', 'psnr_vis', 'psnr_ir', 'edge_intensity']
print(f"{'Metric':<22} {'Average':>10} {'PCA':>10} {'Wavelet':>10}")
print("-" * 55)
for key in keys:
    avg_val = np.mean([m[key] for m in avg_metrics])
    pca_val = np.mean([m[key] for m in pca_metrics])
    dwt_val = np.mean([m[key] for m in dwt_metrics])
    print(f"{key:<22} {avg_val:>10.4f} {pca_val:>10.4f} {dwt_val:>10.4f}")

## Step 7: Visualize results
print("\nGenerating comparison figures...")
visualize_results(pairs, avg_results, pca_results, dwt_results, n_samples=N_SAMPLES)
print("\nPipeline complete.")