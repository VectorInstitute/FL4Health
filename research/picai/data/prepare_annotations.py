import argparse
import os

import SimpleITK as sitk


def prepare_annotations(human_annotations_dir: str, ai_annotations_dir: str, annotations_write_dir: str) -> None:
    """
    Copy separate annotation sources (ie human and ai-derived annotations) to a central location.

    Args:
        human_annotations_dir (str): The path to the folder containing human annotations.
        ai_annotations_dir (str): The path to the folder containing ai-derived annotations.
        annotations_write_dir (str): The path to copy the human and ai-derived annotations to.
    """
    for filename in sorted(os.listdir(human_annotations_dir)):
        path = os.path.join(human_annotations_dir, filename)
        annotation_write_path = os.path.join(annotations_write_dir, filename)
        if path.endswith(".nii.gz"):
            annotation = sitk.ReadImage(path)
            sitk.WriteImage(annotation, str(annotation_write_path), useCompression=True)

    for filename in os.listdir(ai_annotations_dir):
        path = os.path.join(ai_annotations_dir, filename)
        annotation_write_path = os.path.join(annotations_write_dir, filename)
        # In cases where both a human and ai-derived annotation exist
        # We use the human annotation
        if path.endswith(".nii.gz") and not os.path.exists(annotation_write_path):
            annotation = sitk.ReadImage(path)
            sitk.WriteImage(annotation, str(annotation_write_path), useCompression=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    human_annotations_dir_default = (
        "/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/human_expert/resampled"
    )
    ai_annotations_dir_default = (
        "/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/AI/Bosma22a"
    )
    annotations_write_dir_default = (
        "/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/all_annotations_resampled"
    )
    parser.add_argument("--human_annotations_dir", default=human_annotations_dir_default, type=str)
    parser.add_argument(
        "--ai_annotations_dir",
        default=ai_annotations_dir_default,
        type=str,
    )
    parser.add_argument("--annotations_write_dir", default=annotations_write_dir_default, type=str)
    args = parser.parse_args()

    prepare_annotations(args.human_annotations_dir, args.ai_annotations_dir, args.annotations_write_dir)


if __name__ == "__main__":
    main()
