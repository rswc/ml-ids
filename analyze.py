import csv
import river
from river import base

from collections import defaultdict
from river import tree
from tqdm import tqdm

CSV_TEST_FILEPATH = r'cicids2017\all_days.csv'

if __name__ == '__main__':
    
    entry = dict()
    with open(CSV_TEST_FILEPATH, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        
        headers = next(csvreader)
        print(headers)
        print(f"Features: {len(headers)}")
        
        counter = defaultdict(int)
        
        cnt_rows = 0
        for row in tqdm(csvreader):
            # label: index -2 
            if cnt_rows == 0:
                for i, (head, e) in enumerate(zip(headers, row)):
                    print(f"[{i}]: {head} -> {e}")
                    

            cnt_rows += 1
            try:
                counter[row[-2]] += 1
            except:
                print(len(row), row, cnt_rows)
                break
        
        print(f"Samples: {cnt_rows}")
        print("Classes:")
        for i, (key, value) in enumerate(counter.items()):
            print(f"[{i}]: ({value}) -> {key}")



        


