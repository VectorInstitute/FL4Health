# from torchvision.models import swin_v2_t
# from opacus.validators import ModuleValidator

# model = swin_v2_t(weights="IMAGENET1K_V1")

# from research.flamby.utils import summarize_model_info

# summarize_model_info(model)

# for name, module in model.named_modules():
#     print(type(module))

# errors = ModuleValidator.validate(model)
# print(errors)

from flamby.datasets.fed_tcga_brca import metric as c_index
import torch 


y_true = torch.tensor([[   0., 1853.],
        [   0.,  437.],
        [   0.,  666.],
        [   0., 1102.],
        [   0.,  417.],
        [   0.,   10.],
        [   0.,  890.],
        [   0.,  648.],
        [   0.,    5.]])
pred = torch.tensor([[14.4182],
        [10.5080],
        [11.3208],
        [10.2388],
        [13.1592],
        [12.0843],
        [11.3411],
        [ 8.3181],
        [10.2643]])
idx = c_index(y_true, pred)
print(idx)