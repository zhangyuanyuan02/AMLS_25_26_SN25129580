import argparse
from pathlib import Path

from A.model_a import TaskA
from B.model_b import TaskB


def _parse_download_flag(v: str):
    v = v.strip().lower()
    if v in {"auto"}:
        return "auto"
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError("--download must be one of: auto/true/false")


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


def _print_metrics(prefix: str, metrics: dict):
    print(
        f"{prefix} "
        f"Acc={metrics['accuracy']:.4f}, "
        f"P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="AMLS Assignment: BreastMNIST ")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both", help="Which task(s) to run")
    parser.add_argument(
        "--download",
        type=_parse_download_flag,
        default="auto",
        help="Dataset download behavior: auto/true/false auto downloads",
    )

    # Task A controls (capacity/augmentation/budget/features)
    parser.add_argument("--svm_C", type=float, default=1.0, help="Task A: SVM C")
    parser.add_argument("--svm_gamma", type=str, default="scale", help="Task A: SVM gamma")
    parser.add_argument("--a_aug", action="store_true", help="Task A: enable augmentation")
    parser.add_argument("--a_no_aug", action="store_true", help="Task A: disable augmentation")
    parser.add_argument("--a_feature", choices=["raw", "processed"], default="processed", help="Task A: feature pipeline")
    parser.add_argument(
        "--a_budget",
        type=float,
        default=None,
        help="Task A: training budget as fraction of train set (e.g., 0.5 keeps 50%)",
    )

    # Task B controls (capacity/augmentation/budget)
    parser.add_argument("--epochs", type=int, default=20, help="Task B: epochs (training budget)")
    parser.add_argument("--batch_size", type=int, default=64, help="Task B: batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Task B: learning rate")
    parser.add_argument("--b_channels", type=int, default=16, help="Task B: base channels")
    parser.add_argument("--b_aug", action="store_true", help="Task B: enable augmentation")
    parser.add_argument("--b_no_aug", action="store_true", help="Task B: disable augmentation")
    parser.add_argument(
        "--b_budget",
        type=float,
        default=None,
        help="Task B: training budget as fraction of train set (e.g., 0.5 keeps 50%)",
    )

    # Requirement-oriented helpers
    parser.add_argument(
        "--compare_features",
        action="store_true",
        help="Task A only: compare raw vs processed features (re-trains twice)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_root = resolve_dataset_root(project_root)

    # Resolve augmentation flags (default True if neither provided)
    a_aug = True
    if args.a_no_aug:
        a_aug = False
    elif args.a_aug:
        a_aug = True

    b_aug = True
    if args.b_no_aug:
        b_aug = False
    elif args.b_aug:
        b_aug = True

    # ------------------------- Task A -------------------------
    if args.task in {"A", "both"}:
        print("\n[Running Task A: Classical Machine Learning Model]")

        model_a = TaskA(
            dataset_path=str(dataset_root),
            svm_C=args.svm_C,
            svm_gamma=args.svm_gamma,
            use_augmentation=a_aug,
            feature_mode=args.a_feature,
            train_budget=args.a_budget,
            download=args.download,
        )

        if args.compare_features:
            results = model_a.compare_feature_pipelines()
            for mode, metrics in results.items():
                _print_metrics(f"Task A Test ({mode}):", metrics)
        else:
            print("--> Training Model A...")
            model_a.train()
            print("--> Testing Model A...")
            metrics_a = model_a.test()
            _print_metrics("Task A Test:", metrics_a)

    # ------------------------- Task B -------------------------
    if args.task in {"B", "both"}:
        print("\n[Running Task B: Deep Learning Model]")

        model_b = TaskB(
            dataset_path=str(dataset_root),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_augmentation=b_aug,
            base_channels=args.b_channels,
            train_budget=args.b_budget,
            download=args.download,
        )

        print("--> Training Model B...")
        model_b.train()
        print("--> Testing Model B...")
        metrics_b = model_b.test()
        _print_metrics("Task B Test:", metrics_b)

    print("\nAll requested task(s) completed.")


if __name__ == "__main__":
    main()
