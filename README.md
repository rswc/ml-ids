## ML - Intrusion Detection System

- `analyze.py`: extract information from `all_days.csv`
- `example.py`: Example usage with riverml api 

## CICIDS2017

Dataset used inside repository is the improved version of the CICIDS2017 dataset which can be found below: 

- Dataset repo: https://github.com/GintsEngelen/CNS2022_Code
- Dataset download: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip

After downloading dataset, all days can be merged into one dataset using `preprocess_cicids.py`. It also creates `_noatt` version which combines all attempted attacks into **BENIGN** class (which is currently used by `CICIDS2017` class).

All files are by default located inside `cicids2017` directory in project root. This setting can be overriten using `CICIDS2017` class default value or by changing `directory` passed to the `__init__`. 


Default structure of the CICIDS dataset files is shown below

```
cicids2017/
├─ days/
│  ├─ friday.csv
│  ├─ monday.csv
│  ├─ thursday.csv
│  ├─ tuesday.csv
│  ├─ wednesday.csv
├─ all_days.csv
├─ all_days_noatt.csv
├─ all_days_noatt_idx=200_n=4000.csv
├─ all_days_idx=4000_n=4000.csv
```
