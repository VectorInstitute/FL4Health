from enum import Enum
from typing import Any


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
            list[str]: a list containing the names of the attributes for this member
        """
        return self.attribute_keys  # These are set in __new__ for each member

    def values(self) -> list[Any]:
        """
        Gets enum values.

        Returns:
            list[Any]: A list of the attribute values for this member
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
            Any: The main attribute value given a member's attributes.
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
