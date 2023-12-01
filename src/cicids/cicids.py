from river import stream
from river.datasets import base
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class CICIDS2017(base.FileDataset):
    """Class used to handle and load the CICIDS2017 dataset.

    Parameters
    ----------
    n_samples
`       Number of samples in the dataset.
    n_classes
        Number of classes in the dataset. Now works with 16 classes, all attempted attacks are classified as BENIGN.
    n_features
        Number of features in the dataset.
    task
        Type of task the dataset is meant for. Currently only supports multi-class classification.
    filename
        Name of file containing data.
    directory
        The directory where the file is contained.
    """

    features = [
        "id",
        "Flow ID",
        "Src IP",
        "Src Port",
        "Dst IP",
        "Dst Port",
        "Protocol",
        "Timestamp",
        "Flow Duration",
        "Total Fwd Packet",
        "Total Bwd packets",
        "Total Length of Fwd Packet",
        "Total Length of Bwd Packet",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "Fwd RST Flags",
        "Bwd RST Flags",
        "Fwd Header Length",
        "Bwd Header Length",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Packet Length Min",
        "Packet Length Max",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWR Flag Count",
        "ECE Flag Count",
        "Down/Up Ratio",
        "Average Packet Size",
        "Fwd Segment Size Avg",
        "Bwd Segment Size Avg",
        "Fwd Bytes/Bulk Avg",
        "Fwd Packet/Bulk Avg",
        "Fwd Bulk Rate Avg",
        "Bwd Bytes/Bulk Avg",
        "Bwd Packet/Bulk Avg",
        "Bwd Bulk Rate Avg",
        "Subflow Fwd Packets",
        "Subflow Fwd Bytes",
        "Subflow Bwd Packets",
        "Subflow Bwd Bytes",
        "FWD Init Win Bytes",
        "Bwd Init Win Bytes",
        "Fwd Act Data Pkts",
        "Fwd Seg Size Min",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
        "ICMP Code",
        "ICMP Type",
        "Total TCP Flow Time",
        "Label",
        "Attempted Category",
    ]

    classes = [
        'BENIGN',
        'FTP-Patator',
        'SSH-Patator',
        'DoS Slowloris',
        'DoS Slowhttptest',
        'DoS Hulk',
        'DoS GoldenEye',
        'Heartbleed',
        'Web Attack - Brute Force',
        'Infiltration',
        'Infiltration - Portscan',
        'Web Attack - XSS',
        'Web Attack - SQL Injection',
        'Botnet',
        'Portscan',
        'DDoS',
    ]

    converters = {
        "id": int,
        "Flow ID": str,
        "Src IP": str,
        "Src Port": int,
        "Dst IP": str,
        "Dst Port": int,
        "Protocol": int,
        "Timestamp": str,
        "Flow Duration": int,
        "Total Fwd Packet": int,
        "Total Bwd packets": int,
        "Total Length of Fwd Packet": int,
        "Total Length of Bwd Packet": int,
        "Fwd Packet Length Max": int,
        "Fwd Packet Length Min": int,
        "Fwd Packet Length Mean": float,
        "Fwd Packet Length Std": float,
        "Bwd Packet Length Max": int,
        "Bwd Packet Length Min": int,
        "Bwd Packet Length Mean": float,
        "Bwd Packet Length Std": float,
        "Flow Bytes/s": float,
        "Flow Packets/s": float,
        "Flow IAT Mean": float,
        "Flow IAT Std": float,
        "Flow IAT Max": int,
        "Flow IAT Min": int,
        "Fwd IAT Total": int,
        "Fwd IAT Mean": float,
        "Fwd IAT Std": float,
        "Fwd IAT Max": int,
        "Fwd IAT Min": int,
        "Bwd IAT Total": int,
        "Bwd IAT Mean": float,
        "Bwd IAT Std": float,
        "Bwd IAT Max": int,
        "Bwd IAT Min": int,
        "Fwd PSH Flags": int,
        "Bwd PSH Flags": int,
        "Fwd URG Flags": int,
        "Bwd URG Flags": int,
        "Fwd RST Flags": int,
        "Bwd RST Flags": int,
        "Fwd Header Length": int,
        "Bwd Header Length": int,
        "Fwd Packets/s": float,
        "Bwd Packets/s": float,
        "Packet Length Min": int,
        "Packet Length Max": int,
        "Packet Length Mean": float,
        "Packet Length Std": float,
        "Packet Length Variance": float,
        "FIN Flag Count": int,
        "SYN Flag Count": int,
        "RST Flag Count": int,
        "PSH Flag Count": int,
        "ACK Flag Count": int,
        "URG Flag Count": int,
        "CWR Flag Count": int,
        "ECE Flag Count": int,
        "Down/Up Ratio": float,
        "Average Packet Size": float,
        "Fwd Segment Size Avg": float,
        "Bwd Segment Size Avg": float,
        "Fwd Bytes/Bulk Avg": float,
        "Fwd Packet/Bulk Avg": float,
        "Fwd Bulk Rate Avg": float,
        "Bwd Bytes/Bulk Avg": float,
        "Bwd Packet/Bulk Avg": float,
        "Bwd Bulk Rate Avg": float,
        "Subflow Fwd Packets": int,
        "Subflow Fwd Bytes": int,
        "Subflow Bwd Packets": int,
        "Subflow Bwd Bytes": int,
        "FWD Init Win Bytes": int,
        "Bwd Init Win Bytes": int,
        "Fwd Act Data Pkts": int,
        "Fwd Seg Size Min": int,
        "Active Mean": float,
        "Active Std": float,
        "Active Max": int,
        "Active Min": int,
        "Idle Mean": float,
        "Idle Std": float,
        "Idle Max": int,
        "Idle Min": int,
        "ICMP Code": int,
        "ICMP Type": int,
        "Total TCP Flow Time": int,
        "Label": str,
        "Attempted Category": int
    },

    def __init__(self, directory='ABSOLUTE_PATH_TO_DATASET_DIRECTORY', filename='all_days_without_attempted.csv', used_features=None):
        if used_features is not None:
            self.used_features = used_features
            if 'Label' not in self.used_features:
                self.used_features.append('Label')
        else:
            self.used_features = [
                "Flow Duration",
                "Flow Packets/s",
                "Flow IAT Mean",
                "Flow IAT Max",
                "Flow IAT Min",
                "Fwd Packets/s",
                "ACK Flag Count",
                "Subflow Bwd Bytes",
                "FWD Init Win Bytes",
                "Bwd Init Win Bytes",
                "Label",
            ]
        super().__init__(
            n_samples=1787064,
            n_classes=16,  # With all attempted attacks classified as BENIGN
            n_features=len(self.used_features) - 1,  # 90 in total 10 used by default
            task=base.MULTI_CLF,
            filename=filename,
            directory=directory
        )

    def __iter__(self):
        drop_features = [feature for feature in self.features if feature not in self.used_features]

        used_converters = {}
        for key in self.used_features:
            used_converters[key] = self.converters[0][key]

        return stream.iter_csv(
            self.path,
            target="Label",
            converters=used_converters,
            drop=drop_features
        )

    def plot_class_hist(self, window_size: int = 1000):
        classes = self.classes

        result_dict = {element: 0 for element in classes}
        last_samples = []
        class_percentages = []

        for x in tqdm(iter(self)):
            if x[1] not in classes:
                continue

            if len(last_samples) == window_size:
                removed_sample = last_samples.pop(0)
                result_dict[removed_sample] -= 1

            last_samples.append(x[1])
            result_dict[x[1]] += 1

            n_samples = len(last_samples)
            class_percentages.append([((val / n_samples) * 100) for val in result_dict.values()])

        class_percentages = np.array(class_percentages)
        df = pd.DataFrame(class_percentages, columns=classes)

        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(40, 20))
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        sns.lineplot(data=df, palette="tab10", linewidth=2.5)
        plt.show()
