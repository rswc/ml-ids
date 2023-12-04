# Merge 5 days into whole dataset
# - add-up id's 
from pathlib import Path
import csv
from tqdm import tqdm


def merge_cicids(cic_dir, result_csv_name):
    csv_dir = Path(cic_dir)
    csv_result = csv_dir / result_csv_name

    with open(csv_result, 'w', newline='') as csvout:
        csvwriter = csv.writer(csvout)
        last_id = 0
        first_file = True

        for csv_day in tqdm(['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
            csv_filename = csv_dir / f"{csv_day}.csv"

            n_rows = 0

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


def convert_attempted_to_benign(input_file, output_file, label_name):
    with (open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output):
        csv_reader = csv.reader(csv_input)
        csv_writer = csv.writer(csv_output)

        headers = next(csv_reader)
        label_index = headers.index(label_name)
        csv_writer.writerow(headers)

        for row in tqdm(csv_reader):
            if "- Attempted" in row[label_index]:
                row[label_index] = "BENIGN"
            csv_writer.writerow(row)


def create_subset(input_file, output_file, n_samples, start_sample=0):
    with (open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output):
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


if __name__ == '__main__':
    CIC_DIR = '../cicids2017'
    RESULT_CSV_NAME = 'all_days.csv'

    input_data = '../cicids2017/all_days_final.csv'
    final_data = '../cicids2017/graph_test.csv'
    label_col_name = 'Label'

    # merge_cicids(CIC_DIR, RESULT_CSV_NAME)

    # convert_attempted_to_benign(input_data, final_data, label_col_name)

    create_subset(input_data, final_data, 10000, 100000)

                
                

        

