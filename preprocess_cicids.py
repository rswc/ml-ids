from cicids import generate_cicids_file
from pathlib import Path

if __name__ == "__main__":
    dataset_root: Path = None

    cicids_file = generate_cicids_file(
        dataset_path=dataset_root,
        start_sample=400_000,
        n_samples=400_000,
        convert_attempted=True,
    )

    print("Subset generated at: ", cicids_file)
