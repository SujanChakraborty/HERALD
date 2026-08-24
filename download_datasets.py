import os
from pathlib import Path

# import your loader directly from benchmark
from benchmark import load_dataset, ALL_DATASETS

ROOT = "datasets"

def main():
    Path(ROOT).mkdir(exist_ok=True)

    print("Downloading all datasets...\n")

    for name in ALL_DATASETS:
        print(f"=== {name} ===")
        try:
            dataset = load_dataset(name, root=ROOT)
            data = dataset["data"]
            print(f"  ✓ Done: nodes={data.x.shape[0]}, edges={data.edge_index.shape[1]}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nAll datasets processed.")

if __name__ == "__main__":
    main()