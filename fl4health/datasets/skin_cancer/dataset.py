from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
from fl4health.utils.dataset import BaseDataset


class SkinCancerDataset(BaseDataset):
    def __init__(
            self, data: List[Dict[str, Any]],
            transform: Optional[transforms.Compose] = None,
            num_workers: int = 8
            ) -> None:
        super().__init__()
        self.data: List[torch.Tensor] = []
        self.targets: List[int] = []
        self.num_workers = num_workers
        self.transform = transform if transform is not None else transforms.ToTensor()
        self._load_data(data)

    def _load_image(self, item: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        image_path = item["img_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(item["extended_labels"].index(1.0)).item()  # .item() to convert to Python int
        return image, target

    def _load_data(self, data: List[Dict[str, Any]]) -> None:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_image, data))

        self.data, self.targets = zip(*results)
        self.data = torch.stack(list(self.data))  # Convert list of images to a tensor
        self.targets = torch.tensor(self.targets)  # Convert list of targets to a tensor

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        data = self.data[item]
        target = self.targets[item]
        return data, target

    def __len__(self) -> int:
        return len(self.data)
