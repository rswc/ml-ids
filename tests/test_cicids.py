from cicids import CICIDS2017, predict_cicids_filename
from cicids.preprocess import _merge_cicids, _create_subset, _convert_attempted_to_benign
import csv
import pytest
CICIDS_TEST = False


class TestCICIDSPreprocess:
    
    def test_merge_cicids(self, tmp_path):
        days = tmp_path / "tmp-days"
        days.mkdir()
        
        for weekday in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            weekpath = days / f"{weekday}.csv"

            with open(weekpath, 'w', newline='') as dataset_csv:
                csv_writer = csv.writer(dataset_csv)
                csv_writer.writerow(['weekday'])
                for idx in range(50):
                    csv_writer.writerow([str(idx), weekday])

        merged_file = _merge_cicids(dataset_path=tmp_path, days_dir='tmp-days')
        assert merged_file.name == 'all_days.csv' == CICIDS2017.DEFAULT_MERGED_FILENAME

        with open(merged_file) as merged_csv:
            csv_reader = csv.reader(merged_csv)
            assert next(csv_reader)[0] == 'weekday' 
            idx = 0
            for _ in range(50): 
                assert next(csv_reader) == [str(idx), 'monday']
                idx += 1
            for _ in range(50): 
                assert next(csv_reader) == [str(idx), 'tuesday']
                idx += 1
            for _ in range(50): 
                assert next(csv_reader) == [str(idx), 'wednesday']
                idx += 1
            for _ in range(50): 
                assert next(csv_reader) == [str(idx), 'thursday']
                idx += 1
            for _ in range(50): 
                assert next(csv_reader) == [str(idx), 'friday']
                idx += 1
            
            with pytest.raises(StopIteration):
                next(csv_reader)
                
            assert (idx == 250)

        
    def test_convert_attempted_to_benign(self, tmp_path):
        dataset = tmp_path / "all_days.csv"
        assert dataset.name == predict_cicids_filename(False, 0, None)
        
        data = [CICIDS2017.LABEL_COLUMN_NAME] + ['A - Attempted'] * 50 + ['B - Attempted'] * 50

        with open(dataset, 'w', newline='') as dataset_csv:
            csv_writer = csv.writer(dataset_csv)
            for label in data:
                csv_writer.writerow([label])
                
        conv_dataset = _convert_attempted_to_benign(dataset)
        assert conv_dataset.name == "all_days_noatt.csv" == predict_cicids_filename(True, 0, None)

        with open(conv_dataset) as dataset_csv:
            csv_reader = csv.reader(dataset_csv)
            assert next(csv_reader)[0] == 'Label' == CICIDS2017.LABEL_COLUMN_NAME 
            for row in csv_reader:
                assert row[0] == 'BENIGN'
    
    def test_create_subset_converted(self, tmp_path):
        dataset = tmp_path / "all_days_noatt.csv"
        assert dataset.name == predict_cicids_filename(True, 0, None)
        
        data = ['column'] + ['A'] * 50 + ['B'] * 50 + ['C'] * 100 
        with open(dataset, 'w', newline='') as dataset_csv:
            csv_writer = csv.writer(dataset_csv)
            for label in data:
                csv_writer.writerow([label])

        b_only = _create_subset(dataset, start_sample=50, n_samples=50)
        assert b_only.name == 'all_days_noatt_idx=50_n=50.csv' == predict_cicids_filename(True, 50, 50)
        with open(b_only, 'r') as dataset_csv:
            csv_reader = csv.reader(dataset_csv)
            assert next(csv_reader)[0] == 'column'
            for row in csv_reader:
                assert row[0] == 'B'

        bc_only = _create_subset(dataset, start_sample=50, n_samples=None)
        assert bc_only.name == 'all_days_noatt_idx=50_n=None.csv' == predict_cicids_filename(True, 50, None)
        with open(bc_only, 'r') as dataset_csv:
            csv_reader = csv.reader(dataset_csv)
            assert next(csv_reader)[0] == 'column'
            for _ in range(50):
                assert next(csv_reader)[0] == 'B'
            for _ in range(100):
                assert next(csv_reader)[0] == 'C'

    def test_create_subset_attempted(self, tmp_path):
        dataset = tmp_path / "all_days.csv"
        assert dataset.name == predict_cicids_filename(False, 0, None)
        
        data = ['column'] + ['A'] * 50 + ['B'] * 50 + ['C'] * 100 
        with open(dataset, 'w', newline='') as dataset_csv:
            csv_writer = csv.writer(dataset_csv)
            for label in data:
                csv_writer.writerow([label])

        b_only = _create_subset(dataset, start_sample=50, n_samples=50)
        assert b_only.name == 'all_days_idx=50_n=50.csv' == predict_cicids_filename(False, 50, 50)
        with open(b_only, 'r') as dataset_csv:
            csv_reader = csv.reader(dataset_csv)
            assert next(csv_reader)[0] == 'column'
            for row in csv_reader:
                assert row[0] == 'B'

        bc_only = _create_subset(dataset, start_sample=50, n_samples=None)
        assert bc_only.name == 'all_days_idx=50_n=None.csv' == predict_cicids_filename(False, 50, None)
        with open(bc_only, 'r') as dataset_csv:
            csv_reader = csv.reader(dataset_csv)
            assert next(csv_reader)[0] == 'column'
            for _ in range(50):
                assert next(csv_reader)[0] == 'B'
            for _ in range(100):
                assert next(csv_reader)[0] == 'C'

    

class TestCICIDSDataset:
    
    def test_number_of_samples(self):
        if CICIDS_TEST:
            dataset = CICIDS2017()
            samples = 0

            for x, y in iter(dataset):
                samples += 1

            assert samples == dataset.n_samples
        else:
            pass

    def test_number_of_features(self):
        if CICIDS_TEST:
            dataset = CICIDS2017()
            x, y = next(iter(dataset))

            assert len(x) == dataset.n_features
        else:
            pass

    def test_number_of_classes_converted(self):
        if CICIDS_TEST:
            dataset = CICIDS2017(convert_attempted=True)
            assert dataset.filename == 'all_days_noatt.csv'
            classes = []

            for x, y in iter(dataset):
                if y not in classes:
                    classes.append(y)

            assert len(classes) == dataset.n_classes == 16
        else:
            pass

    def test_number_of_classes_converted(self):
        if CICIDS_TEST:
            dataset = CICIDS2017(convert_attempted=False)
            assert dataset.filename == 'all_days.csv'
            classes = []

            for x, y in iter(dataset):
                if y not in classes:
                    classes.append(y)

            assert len(classes) == dataset.n_classes == 27
        else:
            pass
