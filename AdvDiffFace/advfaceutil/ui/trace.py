__all__ = ["trace"]

from functools import wraps
from inspect import signature
from tkinter import Variable
from typing import Literal


def trace(variable_name: str, mode: Literal["array", "read", "write", "unset"]):
    """
    Make the decorated function be the callback for the given variable and mode

    :param variable_name: The name of the variable that this function should trace
    :param mode: The mode of the trace. This must be one of "array", "read", "write" or "unset".
    """

    # We need a class decorator to be able to override the init function
    class TraceDecorator:
        # This is called when the function is defined, passing in the function that is decorated
        def __init__(self, f):
            self.f = f
            # Calculate the function signature
            self.__signature = signature(f)

        def __set_name__(self, owner: type, name: str):
            # This is called when the name of the function is set (this is called automatically)
            # Get the previous __init__ function of the owning class (i.e. the class the decorated function is in)
            old_init = owner.__init__

            # Create a new __init__ method for the owning class which registers the variable at the end
            @wraps(owner.__init__)
            def __init__(this, *args, **kwargs):
                # Call the old initialisation first
                old_init(this, *args, **kwargs)
                # Get the variable we want to trace
                # This handles referencing a property of any variable on this object
                variable_parts = variable_name.split(".")
                variable = this
                for var_name in variable_parts:
                    variable = variable.__dict__[var_name]
                assert isinstance(variable, Variable)

                # Construct a modified version of the function that we will register
                @wraps(self.f)
                def modified_function(*a):
                    # This modified version allows the callback function to not specify all parameters
                    # Normally we would be forced to have three parameters but this handles the case when
                    # we don't specify those and simply ignore them
                    if len(self.__signature.parameters) - 1 < len(a):
                        self.f(this, *(a[: len(self.__signature.parameters) - 1]))
                    else:
                        self.f(this, *a)

                # Trace the variable in the given mode with the modified function
                variable.trace_add(mode, modified_function)

            # Override the initialisation method of the owning class
            owner.__init__ = __init__

    return TraceDecorator
