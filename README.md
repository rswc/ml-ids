## ML - Intrusion Detection System

- Dataset repo: https://github.com/GintsEngelen/CNS2022_Code
- Dataset download: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip

### Info

ALl `.csv` files should be located inside `cicids2017` directory in root of the project.

### Source

- `merge.py`: concatenate 5 days into 1 dataset (`all_days.csv`) 
- `analyze.py`: extract information from `all_days.csv`
- `cicids.py`: RiverML-API adaptation of CICIDS2017 dataset
- `example.py`: Example usage with riverml api 