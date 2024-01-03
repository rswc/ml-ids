from river import stream
from river.datasets import base
from pathlib import Path
from utils import get_project_root


class CICIDS2017(base.FileDataset):
    """Class used to handle and load the CICIDS2017 dataset.

    Parameters
    ----------
    n_samples
        Number of samples in the dataset.
    n_classes
        Number of classes in the dataset. Works with 2 variants 27 and 16 classes (all attempted attacks are classified as BENIGN).
    n_features
        Number of features in the dataset.
    task
        Type of task the dataset is meant for. Currently only supports multi-class classification.
    filename
        Name of file containing data.
    directory
        The directory where the file is contained.
    """

    DEFAULT_DATSET_DIR = "cicids2017"
    DEFAULT_MERGED_FILENAME = "all_days.csv"
    DEFAULT_NOATT_FILENAME = "all_days_noatt.csv"
    LABEL_COLUMN_NAME = "Label"

    class Features:
        """Helper class containing definitions of feature subsets. For passing to dataset via `used_features` param.
        The features are ranked in non-increasing order of importance, so feel free to use Python slicing to select top-N.
        
        """

        YULIANTO2019 = [
            "Total Length of Bwd Packet",
            "Fwd Packet Length Min",
            "Bwd Packet Length Min",
            "Bwd Packet Length Std",
            "Flow IAT Mean",
            "Flow IAT Min",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Min",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Packet Length Min",
            "Packet Length Variance",
            "PSH Flag Count",
            "ACK Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Fwd Segment Size Avg",
            "Subflow Fwd Bytes",
            "Fwd Init Win Bytes",
            "Bwd Init Win Bytes",
            "Active Mean",
            "Idle Mean",
            "Label",
        ]
        """Set of 25 features, as proposed by:
        
        Arif Yulianto et al 2019 J. Phys.: Conf. Ser. 1192 012018,
        "Improving AdaBoost-based Intrusion Detection System (IDS) Performance on CIC IDS 2017 Dataset",
        doi: 10.1088/1742-6596/1192/1/012018
        """

        KURNIABUDI2020 = [
            "Packet Length Std",
            "Total Length of Bwd Packet",
            "Subflow Bwd Bytes",
            "Dst Port",
            "Packet Length Variance",
            "Bwd Packet Length Mean",
            "Bwd Segment Size Avg",
            "Bwd Packet Length Max",
            "Bwd Init Win Bytes",
            "Total Length of Fwd Packet",
            "Subflow Fwd Bytes",
            "Fwd Init Win Bytes",
            "Average Packet Size",
            "Packet Length Mean",
            "Packet Length Max",
            "Fwd Packet Length Max",
            "Flow IAT Max",
            "Bwd Header Length",
            "Flow Duration",
            "Fwd IAT Max",
            "Fwd Header Length",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Flow IAT Mean",
            "Flow Bytes/s",
            "Bwd Packet Length Std",
            "Subflow Bwd Packets",
            "Total Bwd packets",
            "Fwd Packet Length Mean",
            "Fwd Segment Size Avg",
            "Label",
        ]
        """Set of top 30 features according to the ranking proposed by:
        
        Kurniabudi, D. Stiawan, Darmawijoyo, M. Y. Bin Idris, A. M. Bamhdi and R. Budiarto,
        "CICIDS-2017 Dataset Feature Analysis With Information Gain for Anomaly Detection,"
        in IEEE Access, vol. 8, pp. 132911-132921, 2020, doi: 10.1109/ACCESS.2020.3009843
        """

        #TODO: Add docstring w/ source!
        DEFAULT = [
            "Flow Duration",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd Packets/s",
            "ACK Flag Count",
            "Subflow Bwd Bytes",
            "Fwd Init Win Bytes",
            "Bwd Init Win Bytes",
            "Label",
        ]

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
        "Fwd Init Win Bytes",
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

    plain_classes = [
        "BENIGN",
        "FTP-Patator",
        "SSH-Patator",
        "DoS Slowloris",
        "DoS Slowhttptest",
        "DoS Hulk",
        "DoS GoldenEye",
        "Heartbleed",
        "Web Attack - Brute Force",
        "Infiltration",
        "Infiltration - Portscan",
        "Web Attack - XSS",
        "Web Attack - SQL Injection",
        "Botnet",
        "Portscan",
        "DDoS",
    ]

    attempted_classes = [
        "FTP-Patator - Attempted",
        "SSH-Patator - Attempted",
        "DoS Slowloris - Attempted",
        "DoS Slowhttptest - Attempted",
        "DoS Hulk - Attempted",
        "DoS GoldenEye - Attempted",
        "Web Attack - Brute Force - Attempted",
        "Infiltration - Attempted",
        "Web Attack - XSS - Attempted",
        "Web Attack - SQL Injection - Attempted",
        "Botnet - Attempted",
    ]

    converters = (
        {
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
            "Fwd Init Win Bytes": int,
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
            "Attempted Category": int,
        },
    )

    def __init__(
        self,
        filename: str = None,
        dataset_dir: Path = None,
        used_features: list = None,
        convert_attempted: bool = True,
        n_samples: int = None,
    ):
        if filename is None:
            filename = [
                CICIDS2017.DEFAULT_MERGED_FILENAME,
                CICIDS2017.DEFAULT_NOATT_FILENAME,
            ][convert_attempted]
        if used_features is not None:
            self.used_features = used_features
            if self.LABEL_COLUMN_NAME not in self.used_features:
                self.used_features.append(self.LABEL_COLUMN_NAME)
        else:
            self.used_features = CICIDS2017.Features.DEFAULT

        directory = (dataset_dir or self.default_dataset_dir()).resolve()
        if not directory.is_dir():
            raise ValueError(
                f"Specified directory '{directory}' does not exist or is not a valid directory."
            )

        dataset_filepath = directory / filename
        if not dataset_filepath.is_file():
            raise ValueError(
                f"Specified dataset file '{dataset_filepath}' does not exist or is not a valid file."
            )

        if convert_attempted:
            # With all attempted attacks classified as BENIGN
            self.classes = CICIDS2017.plain_classes
        else:
            # Otherwise add all Attempted classes
            self.classes = CICIDS2017.plain_classes + CICIDS2017.attempted_classes

        super().__init__(
            n_samples=n_samples or 2099976,  # Samples for subset or full dataset
            n_classes=len(self.classes),
            # -1 because label is not a feature in stream
            n_features=len(self.used_features) - 1,
            task=base.MULTI_CLF,
            filename=filename,
            directory=str(directory),
        )

    @staticmethod
    def default_dataset_dir() -> Path:
        return get_project_root() / CICIDS2017.DEFAULT_DATSET_DIR

    def __iter__(self):
        drop_features = [
            feature for feature in self.features if feature not in self.used_features
        ]

        used_converters = {}
        for key in self.used_features:
            used_converters[key] = self.converters[0][key]

        return stream.iter_csv(
            self.path,
            target=self.LABEL_COLUMN_NAME,
            converters=used_converters,
            drop=drop_features,
        )
