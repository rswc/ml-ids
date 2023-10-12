from river import stream
from river.datasets import base

class CICIDS2017(base.FileDataset):
    """TODO: title

    TODO: descr

    References
    ----------
    [^1] []()

    """

    def __init__(self):
        super().__init__(
            n_samples=2099976,
            n_classes=27,
            n_features=91,
            task=base.MULTI_CLF,
            filename="all_days.csv",
            directory="cicids2017"
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="Label",
            converters={
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
                # TODO: Check below
                "Fwd Bytes/Bulk Avg": float,
                "Fwd Packet/Bulk Avg": float,
                "Fwd Bulk Rate Avg": float,
                "Bwd Bytes/Bulk Avg": float,
                "Bwd Packet/Bulk Avg": float,
                "Bwd Bulk Rate Avg": float,
                ### 
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
                "Attempted Category": int,
            },
        )
