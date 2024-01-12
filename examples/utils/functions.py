from typing import Optional


def make_dict_with_epochs_or_steps(
    local_epochs: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> dict[str, int]:
    if local_epochs is not None:
        return {"local_epochs": local_epochs}
    if local_steps is not None:
        return {"local_steps": local_steps}
    else:
        return {}
