import os
import sys
from A.model_a import TaskA
from B.model_b import TaskB

def main():

    dataset_path = os.path.join(os.getcwd(), 'Datasets')
    
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset folder not found at: {dataset_path}")
        print("Please ensure 'Datasets' folder exists in the project root.")
        return

    # Model A
    print("\n[Running Task A: Classical Machine Learning Model]")

    model_a = TaskA(dataset_path=dataset_path)
    
    print("--> Training Model A...")
    acc_a = model_a.train()
    
    # Execute testing for Model A
    print("--> Testing Model A...")
    test_score_a = model_a.test()
    print(f"Task A Finished. Test Accuracy: {test_score_a:.4f}")

    # Model B
    print("\n[Running Task B: Deep Learning Model]")
    model_b = TaskB(dataset_path=dataset_path)
    
    print("--> Training Model B...")
    acc_b = model_b.train()
    
    print("--> Testing Model B...")
    test_score_b = model_b.test()
    print(f"Task B Finished. Test Accuracy: {test_score_b:.4f}")
    
    print("\n==================================================")
    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()