__all__ = [
    "DataHolder",
]

from typing import Any, Dict

from advfaceutil.benchmark.data.property import DataProperty


class DataHolder:
    def __init__(self):
        self._properties: Dict[DataProperty, Any] = {}

    def add_property(self, key: DataProperty, value: Any) -> None:
        """
        Add a property to the data.

        :param key: The name of the property.
        :param value: The value of the property.
        """
        self._properties[key] = value

    def remove_property(self, key: DataProperty) -> None:
        """
        Remove a property from the data.

        :param key: The name of the property to remove.
        """
        del self._properties[key]

    def has_property(self, key: DataProperty) -> bool:
        """
        Check if the data has a property.

        :param key: The name of the property to check.
        :return: Whether the data has the property.
        """
        return key in self._properties.keys()

    def get_property(self, key: DataProperty) -> Any:
        """
        Get the value of a property.

        :param key: The name of the property to get.
        :return: The value of the property.
        """
        return self._properties[key]

    def copy_from(self, other: "DataHolder") -> None:
        self._properties.update(other._properties)
