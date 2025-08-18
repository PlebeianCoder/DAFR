__all__ = ["EnumVar", "EnumOptionMenu"]

from enum import Enum
from tkinter import StringVar
from tkinter.ttk import OptionMenu
from typing import Optional, Type
from typing import TypeVar
from typing import Union


E = TypeVar("E", bound=Enum)


class EnumVar(StringVar):
    def __init__(self, master, enum: Type[E], value: E, name: Optional[str] = None):
        super().__init__(master, value.name, name)
        self.__enum = enum

    def set(self, value: Union[str, E]):
        # If the value is not an instance of the enum or is not a member of the enum
        # then raise an error
        if not isinstance(value, self.__enum) and (
            isinstance(value, str) and value not in self.__enum.__members__
        ):
            raise ValueError(f"Value {value} is not a member of the enum {self.__enum}")
        # If the value is a string then convert it to the enum value
        if isinstance(value, str):
            value = self.__enum[value]
        # Store the name of the value
        super().set(value.name)

    def get(self) -> E:
        value = super().get()
        return self.__enum[value]


class EnumOptionMenu(OptionMenu):
    def __init__(self, master, variable, enum: Type[Enum], **kwargs):
        super().__init__(
            master,
            variable,
            variable.get(),
            *list(map(lambda x: x.name, enum)),
            **kwargs,
        )
