import os
from enum import Enum
from logging import ERROR, INFO
from pathlib import Path
from typing import Any

import torch
from flwr.common.logger import log


class SimpleDictionaryCheckpointer:
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str,
    ) -> None:
        """
        A simple state checkpointer that saves and loads an object's attribute state (stored in a dictionary) to and
        from a file.

        Args:
            checkpoint_dir (Path): Directory to which checkpoints are saved
            checkpoint_name (str): Name of the checkpoint to be saved
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

    def save_checkpoint(self, checkpoint_dict: dict[str, Any]) -> None:
        """
        Save ``checkpoint_dict`` to checkpoint path defined based on checkpointer dir and checkpoint name.

        Args:
            checkpoint_dict (dict[str, Any]): A dictionary with string keys and values of type Any representing the
                state to be saved.

        Raises:
            e: Will throw an error if there is an issue saving the model. ``Torch.save`` seems to swallow errors in
                this context, so we explicitly surface the error with a try/except.
        """
        try:
            log(INFO, f"Saving the state as {self.checkpoint_path}")
            torch.save(checkpoint_dict, self.checkpoint_path)
        except Exception as e:
            log(ERROR, f"Encountered the following error while saving the checkpoint: {e}")
            raise e

    def load_checkpoint(self) -> dict[str, Any]:
        """
        Load and return the checkpoint stored in ``checkpoint_dir`` under the  ``checkpoint_name`` if it exists. If
        it does not exist, an assertion error will be thrown.

        Returns:
            (dict[str, Any]): A dictionary representing the checkpointed state, as loaded by ``torch.load``.
        """
        assert self.checkpoint_exists()
        log(INFO, f"Loading state from checkpoint at {self.checkpoint_path}")

        return torch.load(self.checkpoint_path, weights_only=False)

    def checkpoint_exists(self) -> bool:
        """
        Check if a checkpoint exists at the ``checkpoint_path`` constructed as ``checkpoint_dir`` +
        ``checkpoint_name`` during initialization.

        Returns:
            (bool): True if checkpoint exists, otherwise false.
        """
        return os.path.exists(self.checkpoint_path)


class MultiAttributeEnum(Enum):
    def __init__(self, attributes: Any) -> None:
        """
        A subclass of Enum that allows members to have multiple attributes. Members can be defined by either
        dictionaries or lists. The attributes of the members must also be defined in the subclass to avoid mypy issues.

        Example Usage:
            Animals(MultiAttributeEnum):
                # Define Attributes
                species: str
                mammal: bool
                height: float
                # Define Members
                Dog = {'species': 'Canis Lupus Familiaris', 'mammal': True, 'height': 0.5}
                Giraffe = {'species': 'Giraffa', 'mammal': True, 'height': 10}

        By default the main attribute is the first value in the dictionary. So
        for the above example:

        >>> Animals("Canis Lupus Familiaris")
        Animals.Dog

        The main attribute can be changed by overriding
        self.get_main_value(). If the value of the main attribute is not
        unique to a specific member, then the member that was defined first is
        returned.

        Members can also be defined using lists so long as the
        self.get_attribute_keys() method is also defined by the user.
        Additionally, not all members need to have all the attributes. This is
        also true when the members are defined using dictionaries. You can even
        mix dictionaries and lists.
        Example Usage:
            Animals(MultiAttributeEnum):
                # Define Attributes
                species: str
                mammal: bool
                height: int

                # Specify attribute keys so we can define members with list
                def get_attribute_keys(self, attributes):
                    return ['species', 'mammal', 'height']

                # Define Members
                Dog = ['Canis Lupus Familiaris', True] # Dog will be missing the height attribute
                Giraffe = {'species': 'Giraffa', 'mammal': True, 'height': 10}
                Cat = ['Felis Catus', True, 0.25]

        Args:
            attributes (dict[str, Any] | List): A list or dictionary of
                attribute values for the enum member. If a list is given then
                self.get_attribute_keys must be defined so that the class
                knows what to name the attributes.
        """
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                setattr(self, key, value)
        elif isinstance(attributes, list):
            attribute_keys = self.get_attribute_keys(attributes)
            for key, value in zip(attribute_keys, attributes):
                setattr(self, key, value)

        # Create attributes that will be assigned for each member separately
        self.attribute_keys: list[str]
        self.attribute_values: list[Any]

    def __new__(cls, attributes: Any) -> Enum:  # type: ignore
        """
        Creates a new member. There is some Enum Weirdness here which is why we
        have to tell mypy to ignore the typing. cls ends up being type
        type[MultiAttributeEnum] but class methods expect type
        MultiAttributeEnum for self. Additionally the return type is expected
        to be MultiAttributeEnum but we can't specify that right now because
        the class hasn't been instantiated. There might be a way around this
        using bound TypeVars and generics but I don't know enough to figure it
        out right now.
        """
        # Create member
        obj = object.__new__(cls)

        # Set the attribute keys, values and main value for the member
        if isinstance(attributes, dict):
            obj._value_ = cls.get_main_value(cls, attributes)  # type: ignore
            obj.attribute_values = list(attributes.values())
            obj.attribute_keys = list(attributes.keys())
        if isinstance(attributes, list):
            obj._value_ = cls.get_main_value(cls, attributes)  # type: ignore
            obj.attribute_values = attributes
            obj.attribute_keys = cls.get_attribute_keys(cls, attributes)[: len(attributes)]  # type: ignore
        else:  # Assume we only have a single attribute
            obj.attribute_values = [attributes]
            obj._value_ = attributes  # reset the main value

        return obj  # Return member

    def keys(self) -> list[str]:
        """
        Gets the names of the enum attributes.

        Returns:
            (list[str]): a list containing the names of the attributes for this member
        """
        return self.attribute_keys  # These are set in __new__ for each member

    def values(self) -> list[Any]:
        """
        Gets enum values.

        Returns:
            (list[Any]): A list of the attribute values for this member
        """
        return self.attribute_values

    def get_main_value(self, attributes: Any) -> Any:
        """
        Returns the main attribute value given a member's attributes. Whatever is returned is what will be used to
        identify the member when calling the class. The main value can be accessed using member.value.

        Example:
            Animals(MultiAttributeEnum):
                # Define Attributes
                species: str
                mammal: bool
                height: int

                # Define Members
                Dog = {'species': 'Canis Lupus Familiaris', 'mammal': True, 'height': 0.5}
                Giraffe = {'species': 'Giraffa', 'mammal': True, 'height': 10}

                def (self, attributes):
                    return attributes['height']

            >>> Animals(10)
            Animals.Giraffe
            >>> Animals(0.5)
            Animals.Dog
            >>> Animals.Dog.value
            0.5

        Args:
            attributes (Any): The attributes used to define the members

        Returns:
            (Any): The main attribute value given a member's attributes.
        """
        if isinstance(attributes, dict):
            return list(attributes.values())[0]
        if isinstance(attributes, list):
            return attributes[0]
        raise ValueError("attributes has an unrecognized type")

    def get_attribute_keys(self, attributes: list) -> list[str]:
        raise NotImplementedError(
            "Received a list of attributes but the self.get_attribute_keys class method was not implemented"
        )
