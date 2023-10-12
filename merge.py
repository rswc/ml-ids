# Merge 5 days into whole dataset
# - add-up id's 
from pathlib import Path
import csv
from tqdm import tqdm

# location of five .csv files (mon - fri)
CIC_DIR = 'cicids2017'
RESULT_CSV_NAME = 'all_days.csv'

if __name__ == '__main__':
    csv_dir = Path(CIC_DIR)
    
    csv_result = csv_dir / RESULT_CSV_NAME
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


                
                

        

