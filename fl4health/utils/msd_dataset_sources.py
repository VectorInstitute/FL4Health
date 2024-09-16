from enum import Enum


class MsdDataset(Enum):
    TASK01_BRAINTUMOUR = "Task01_BrainTumour"
    TASK02_HEART = "Task02_Heart"
    TASK03_LIVER = "Task03_Liver"
    TASK04_HIPPOCAMPUS = "Task04_Hippocampus"
    TASK05_PROSTATE = "Task05_Prostate"
    TASK06_LUNG = "Task06_Lung"
    TASK07_PANCREAS = "Task07_Pancreas"
    TASK08_HEPATICVESSEL = "Task08_HepaticVessel"
    TASK09_SPLEEN = "Task09_Spleen"
    TASK10_COLON = "Task10_Colon"


def get_msd_dataset_enum(dataset_name: str) -> MsdDataset:
    try:
        return MsdDataset(dataset_name)
    except Exception as e:
        raise e


msd_urls = {
    MsdDataset.TASK01_BRAINTUMOUR: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    MsdDataset.TASK02_HEART: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
    MsdDataset.TASK03_LIVER: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
    MsdDataset.TASK04_HIPPOCAMPUS: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
    MsdDataset.TASK05_PROSTATE: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
    MsdDataset.TASK06_LUNG: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
    MsdDataset.TASK07_PANCREAS: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
    MsdDataset.TASK08_HEPATICVESSEL: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
    MsdDataset.TASK09_SPLEEN: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    MsdDataset.TASK10_COLON: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
}

msd_md5_hashes = {
    MsdDataset.TASK01_BRAINTUMOUR: "240a19d752f0d9e9101544901065d872",
    MsdDataset.TASK02_HEART: "06ee59366e1e5124267b774dbd654057",
    MsdDataset.TASK03_LIVER: "a90ec6c4aa7f6a3d087205e23d4e6397",
    MsdDataset.TASK04_HIPPOCAMPUS: "9d24dba78a72977dbd1d2e110310f31b",
    MsdDataset.TASK05_PROSTATE: "35138f08b1efaef89d7424d2bcc928db",
    MsdDataset.TASK06_LUNG: "8afd997733c7fc0432f71255ba4e52dc",
    MsdDataset.TASK07_PANCREAS: "4f7080cfca169fa8066d17ce6eb061e4",
    MsdDataset.TASK08_HEPATICVESSEL: "641d79e80ec66453921d997fbf12a29c",
    MsdDataset.TASK09_SPLEEN: "410d4a301da4e5b2f6f86ec3ddba524e",
    MsdDataset.TASK10_COLON: "bad7a188931dc2f6acf72b08eb6202d0",
}

# The number of classes for each MSD Dataset (including background)
# I got these from the paper, didn't download all the datasets to double check
msd_num_labels = {
    MsdDataset.TASK01_BRAINTUMOUR: 4,
    MsdDataset.TASK02_HEART: 2,
    MsdDataset.TASK03_LIVER: 3,
    MsdDataset.TASK04_HIPPOCAMPUS: 3,
    MsdDataset.TASK05_PROSTATE: 3,
    MsdDataset.TASK06_LUNG: 2,
    MsdDataset.TASK07_PANCREAS: 3,
    MsdDataset.TASK08_HEPATICVESSEL: 3,
    MsdDataset.TASK09_SPLEEN: 2,
    MsdDataset.TASK10_COLON: 2,
}
