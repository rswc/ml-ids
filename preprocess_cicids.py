# Merge 5 days into whole dataset
# - add-up id's 
from pathlib import Path
import csv
from tqdm import tqdm
from cicids import CICIDS2017
import sys

DAYS_DIR = 'days'
DATASET_PATH = Path(CICIDS2017.DEFAULT_DATSET_DIR)

def merge_cicids(dataset_path: Path = DATASET_PATH, days_dir: str = DAYS_DIR, 
                 merged_filename: str = CICIDS2017.DEFAULT_MERGED_FILENAME, force: bool = False):
    """Merge day-related csv's into combined CICIDS2017 dataset csv"""
    
    days_path = dataset_path / days_dir
    merged_path = dataset_path / merged_filename

    assert days_path.is_dir(), f"Directory {days_path} does not exist."
    
    if merged_path.is_file():
        if not force:
            print(f"[!] {merged_path} already exists, use `force=True` to override. Skipping...")
            return merged_path

    with open(merged_path, 'w', newline='') as csvout:
        csvwriter = csv.writer(csvout)
        last_id = 0
        first_file = True

        for weekday in tqdm(['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
            csv_filename = days_path / f"{weekday}.csv"
            n_rows = 0
            
            assert csv_filename.is_file(), f"File {csv_filename} does not exist!"

            with open(csv_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)

                headers = next(csv_reader)

                if first_file:
                    csvwriter.writerow(headers)
                    first_file = False

                for row in csv_reader:
                    row_id, *rest = row
                    row_id = str(int(row_id) + last_id)
                    new_row = [row_id] + rest
                    csvwriter.writerow(new_row)
                    n_rows += 1
            last_id += n_rows
    return merged_path


def convert_attempted_to_benign(input_path: Path, output_path: Path = None, label_name = CICIDS2017.LABEL_COLUMN_NAME, force: bool = False):
    if output_path is None:
        output_path = input_path.with_stem(f"{input_path.stem}_noatt")
        
    if output_path.is_file():
        if not force:
            print(f"[!] {output_path} already exists, use `force=True` to override. Skipping...", file=sys.stderr)
            return output_path
        
        output_path.unlink()

    with (
        open(input_path, 'r') as csv_input, 
        open(output_path, 'w', newline='') as csv_output
    ):
        csv_reader = csv.reader(csv_input)
        csv_writer = csv.writer(csv_output)

        headers = next(csv_reader)
        label_index = headers.index(label_name)
        csv_writer.writerow(headers)

        for row in tqdm(csv_reader):
            if "- Attempted" in row[label_index]:
                row[label_index] = "BENIGN"
            csv_writer.writerow(row)
            
    return output_path


def create_subset(input_path: Path, n_samples: int, start_sample: int = 0, output_path: Path = None, force: bool = False):
    if output_path is None:
        output_path = input_path.with_stem(f"{input_path.stem}_idx={start_sample}_n={n_samples}")

    if output_path.is_file():
        if not force:
            print(f"[!] {output_path} already exists, use `force=True` to override. Skipping...", file=sys.stderr)
            return output_path
        
        output_path.unlink()

    with (
        open(input_path, 'r') as csv_input,
        open(output_path, 'w', newline='') as csv_output
    ):
        csv_reader = csv.reader(csv_input)
        csv_writer = csv.writer(csv_output)

        headers = next(csv_reader)
        csv_writer.writerow(headers)

        for i, row in enumerate(tqdm(csv_reader)):
            if i < start_sample:
                continue
            else:
                if i >= start_sample + n_samples:
                    break
                csv_writer.writerow(row)

    return output_path


if __name__ == '__main__':
    merged_path = merge_cicids()
    noatt_path = convert_attempted_to_benign(input_path=merged_path)
    subset_path = create_subset(input_path=noatt_path, start_sample=400_000, n_samples=400_000)
    print("Subset generated at: ", subset_path)
