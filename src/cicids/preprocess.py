from cicids import CICIDS2017
from pathlib import Path
from tqdm import tqdm
import csv
import sys

DAYS_DIR = 'days'

def merge_cicids(dataset_path: Path = None, days_dir: str = DAYS_DIR, 
                 merged_filename: str = CICIDS2017.DEFAULT_MERGED_FILENAME, force: bool = False) -> Path:
    """Merge day-related csv's into combined CICIDS2017 dataset csv"""
    
    dataset_path = (dataset_path or CICIDS2017.default_dataset_dir()).resolve()
    days_path = dataset_path / days_dir
    merged_path = dataset_path / merged_filename

    assert days_path.is_dir(), f"Directory {days_path} does not exist."
    
    if merged_path.is_file():
        if not force:
            print(f"[!] {merged_path} already exists, use `force=True` to override. Skipping...", file=sys.stderr)
            return merged_path
        
        merged_path.unlink()

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


def convert_attempted_to_benign(input_path: Path, output_path: Path = None, 
                                label_name = CICIDS2017.LABEL_COLUMN_NAME, force: bool = False) -> Path:
    """ Convert all classes with 'Attempted' suffix to 'BENIGN' """
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


def create_subset(input_path: Path, start_sample: int , n_samples: int, 
                  output_path: Path = None, force: bool = False) -> Path:
    """ Create subset of .csv datset starting with `start_sample` index and collecting `n_samples` of examples """

    if output_path is None:
        output_path = input_path.with_stem(f"{input_path.stem}_idx={start_sample}_n={n_samples}")

    if output_path.is_file():
        if not force:
            print(f"[!] {output_path} already exists, use `force=True` to override. Skipping...", file=sys.stderr)
            return output_path
        
        output_path.unlink()
        
    cnt_samples = 0
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
                if n_samples is not None and i >= start_sample + n_samples:
                    break
                csv_writer.writerow(row)
                cnt_samples += 1
                
    if n_samples is not None and cnt_samples < n_samples:
        print(f"[!] subset generation processed {cnt_samples} which is smaller than specified {n_samples}", file=sys.stderr)

    return output_path

def generate_cicids_file(dataset_path: Path = None, start_sample: int = 0, convert_attempted: bool = True,
                         n_samples: int = None, days_dir: str = DAYS_DIR, force: bool = False) -> Path:
    """ Merge, Convert and return CICIDS subset according to given parameters. Setting n_samples = None collects rest of the dataset"""

    file_path = merge_cicids(dataset_path, days_dir, force=force)
    if convert_attempted:
        file_path = convert_attempted_to_benign(input_path=file_path, force=force)

    if not (start_sample == 0 and n_samples is None):
        file_path = create_subset(input_path=file_path, start_sample=start_sample, n_samples=n_samples, force=force)
    return file_path

def predict_cicids_filename(convert_attempted: bool, start_sample: int, n_samples: int) -> str:
    """ Helper function to make sure dataset path is compliant with the convention """

    suffix = '' if (start_sample == 0 and n_samples is None) else f'_idx={start_sample}_n={n_samples}'
    return f"all_days{['', '_noatt'][convert_attempted]}{suffix}.csv"