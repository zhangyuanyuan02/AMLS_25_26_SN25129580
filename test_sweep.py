import argparse
import os
import random
import sys
from pathlib import Path
import itertools

import numpy as np
import torch

# Add current directory to sys.path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parent))

from A.model_a import TaskA
from B.model_b import TaskB


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_dataset_root(project_root: Path) -> Path:
    datasets_dir = project_root / "Datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    root_a = datasets_dir / "BreastMNIST"
    root_b = datasets_dir

    if (root_a / "breastmnist.npz").exists():
        return root_a
    if (root_b / "breastmnist.npz").exists():
        return root_b

    # allow online download into a clean folder
    root_a.mkdir(parents=True, exist_ok=True)
    return root_a

def run_task_a_experiments(dataset_path, results_file, seed: int):
    print("\n" + "="*30)
    print("Running Task A Experiments (SVM)")
    print("="*30)
    
    results_file.write("\n" + "="*30 + "\n")
    results_file.write("Running Task A Experiments (SVM)\n")
    results_file.write("="*30 + "\n")

    # Parameters to vary
    svm_Cs = [0.1, 1.0, 10.0]
    feature_modes = ["raw", "processed"]
    augmentations = [False, True]
    
    # Iterate over all combinations (use validation for comparison)
    best_key = None
    best_metrics = None
    best_score = -1.0
    for C, feat_mode, aug in itertools.product(svm_Cs, feature_modes, augmentations):
        print(f"\nExperiment: C={C}, feature_mode={feat_mode}, augmentation={aug}")
        results_file.write(f"\nExperiment: C={C}, feature_mode={feat_mode}, augmentation={aug}\n")
        
        task = TaskA(
            dataset_path=dataset_path,
            svm_C=C,
            feature_mode=feat_mode,
            use_augmentation=aug,
            download="auto",
            random_state=seed,
        )
        
        task.train()
        metrics = task.validate()
        
        result_str = (f"  Val Metrics: Acc={metrics['accuracy']:.4f}, "
                      f"P={metrics['precision']:.4f}, "
                      f"R={metrics['recall']:.4f}, "
                      f"F1={metrics['f1']:.4f}")
        print(result_str)
        results_file.write(result_str + "\n")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_key = (C, feat_mode, aug)
            best_metrics = metrics

    if best_key is not None:
        C, feat_mode, aug = best_key
        print(f"\nBest Task A config by val F1: C={C}, feature_mode={feat_mode}, augmentation={aug}")
        results_file.write(
            f"\nBest Task A config by val F1: C={C}, feature_mode={feat_mode}, augmentation={aug}\n"
        )
        results_file.write(
            f"  Best Val Metrics: Acc={best_metrics['accuracy']:.4f}, "
            f"P={best_metrics['precision']:.4f}, "
            f"R={best_metrics['recall']:.4f}, "
            f"F1={best_metrics['f1']:.4f}\n"
        )

        final_task = TaskA(
            dataset_path=dataset_path,
            svm_C=C,
            feature_mode=feat_mode,
            use_augmentation=aug,
            download="auto",
            random_state=seed,
        )
        final_task.train()
        final_metrics = final_task.test()
        final_str = (f"  Final Test Metrics: Acc={final_metrics['accuracy']:.4f}, "
                     f"P={final_metrics['precision']:.4f}, "
                     f"R={final_metrics['recall']:.4f}, "
                     f"F1={final_metrics['f1']:.4f}")
        print(final_str)
        results_file.write(final_str + "\n")

def run_task_b_experiments(dataset_path, results_file, seed: int):
    print("\n" + "="*30)
    print("Running Task B Experiments (CNN)")
    print("="*30)
    
    results_file.write("\n" + "="*30 + "\n")
    results_file.write("Running Task B Experiments (CNN)\n")
    results_file.write("="*30 + "\n")

    # Parameters to vary
    # Reduced set to keep runtime reasonable for a test script
    lrs = [1e-3, 1e-4]
    base_channels_list = [8, 16]
    augmentations = [False, True]
    epochs = 20 # Reduced epochs for quick testing, can be increased
    
    best_key = None
    best_metrics = None
    best_score = -1.0
    for lr, channels, aug in itertools.product(lrs, base_channels_list, augmentations):
        print(f"\nExperiment: lr={lr}, base_channels={channels}, augmentation={aug}, epochs={epochs}")
        results_file.write(f"\nExperiment: lr={lr}, base_channels={channels}, augmentation={aug}, epochs={epochs}\n")
        
        task = TaskB(
            dataset_path=dataset_path,
            epochs=epochs,
            lr=lr,
            base_channels=channels,
            use_augmentation=aug,
            download="auto",
            seed=seed,
        )
        
        task.train()
        metrics = task.validate()
        
        result_str = (f"  Val Metrics: Acc={metrics['accuracy']:.4f}, "
                      f"P={metrics['precision']:.4f}, "
                      f"R={metrics['recall']:.4f}, "
                      f"F1={metrics['f1']:.4f}")
        print(result_str)
        results_file.write(result_str + "\n")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_key = (lr, channels, aug)
            best_metrics = metrics

    if best_key is not None:
        lr, channels, aug = best_key
        print(
            f"\nBest Task B config by val F1: lr={lr}, base_channels={channels}, augmentation={aug}, epochs={epochs}"
        )
        results_file.write(
            f"\nBest Task B config by val F1: lr={lr}, base_channels={channels}, augmentation={aug}, epochs={epochs}\n"
        )
        results_file.write(
            f"  Best Val Metrics: Acc={best_metrics['accuracy']:.4f}, "
            f"P={best_metrics['precision']:.4f}, "
            f"R={best_metrics['recall']:.4f}, "
            f"F1={best_metrics['f1']:.4f}\n"
        )

        final_task = TaskB(
            dataset_path=dataset_path,
            epochs=epochs,
            lr=lr,
            base_channels=channels,
            use_augmentation=aug,
            download="auto",
            seed=seed,
        )
        final_task.train()
        final_metrics = final_task.test()
        final_str = (f"  Final Test Metrics: Acc={final_metrics['accuracy']:.4f}, "
                     f"P={final_metrics['precision']:.4f}, "
                     f"R={final_metrics['recall']:.4f}, "
                     f"F1={final_metrics['f1']:.4f}")
        print(final_str)
        results_file.write(final_str + "\n")

def main():
    parser = argparse.ArgumentParser(description="AMLS Assignment: Experiment Sweeps")
    parser.add_argument("--seed", type=int, default=25129580, help="Global random seed")
    args = parser.parse_args()

    set_global_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    dataset_root = resolve_dataset_root(project_root)
    results_path = project_root / "experiment_results.txt"
    
    with open(results_path, "w") as f:
        run_task_a_experiments(dataset_root, f, args.seed)
        run_task_b_experiments(dataset_root, f, args.seed)
        
    print(f"\nAll experiments completed. Results saved to {results_path}")

if __name__ == "__main__":
    main()
