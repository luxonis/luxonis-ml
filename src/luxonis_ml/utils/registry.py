from typing import Optional, Callable, Dict, Union, Tuple
from abc import ABCMeta


class Registry:
    def __init__(self, name: str):
        """A registry to map strings to classes or functions.

        Args:
            name (str): Registry name
        """
        self._module_dict: Dict[str, type] = dict()
        self._name = name

    def __str__(self):
        string = f"--- Registry `{self.name}` ---\n"
        for obj_name, obj_cls in self._module_dict.items():
            string += f"{obj_name} {str(obj_cls)}\n"
        string += "------"
        return string

    def __len__(self):
        return len(self._module_dict)

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key: str) -> type:
        """Get the registry record for the key.

        Args:
            key (str): Name of the registered item, e.g. the class name in string format

        Returns:
            tpye: Corresponding class if `key` exists
        """
        module_cls = self._module_dict.get(key, None)
        if module_cls is None:
            raise KeyError(f"Class `{key}` not in the `{self.name}` registry.")
        else:
            return module_cls

    def register_module(
        self,
        name: Optional[str] = None,
        module: Optional[type] = None,
        force: bool = False,
    ) -> Union[type, Callable[[type], type]]:
        """Registers a module.

        Args:
            name (Optional[str], optional): Name of the module, if None then use class name. Defaults to None.
            module (Optional[type], optional): Module class to be registered. Defaults to None.
            force (bool, optional): Wheather to override an existing class with the same name. Defaults to False.

        Returns:
            Union[type, Callable[[type], type]]: Module class or register function if used as a decorator
        """
        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module: type) -> type:
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    def _register_module(
        self, module: type, module_name: Optional[str] = None, force: bool = False
    ) -> None:
        """Registers a module by creating a (key, value) pair.

        Args:
            module (type): Module class to be registered
            module_name (Optional[str], optional): Name of the module, if None use class name. Defaults to None.
            force (bool, optional): Weather to override an existing class with the same name. Defaults to False.

        Raises:
            KeyError: Raised if class name already exists and force==False
        """
        if module_name is None:
            module_name = module.__name__

        if not force and module_name in self._module_dict:
            existed_module = self._module_dict[module_name]
            raise KeyError(
                f"`{module_name}` already registred in `{self.name}` registry at `{existed_module.__module__}`."
            )

        self._module_dict[module_name] = module


class AutoRegisterMeta(ABCMeta):
    """Metaclass for automatically registering modules.

    Can be set as a metaclass for abstract base classes. Then, all subclasses will be
    automatically registered under the name of the subclass.

    Attributes:
        REGISTRY (Registry): Registry used for registration of sub-classes.

    Example:
        >>> REGISTRY = Registry(name="modules")
        >>> class BaseClass(metaclass=AutoRegisterMeta, registry=REGISTRY):
        ...     pass
        >>> class SubClass(BaseClass):
        ...     pass
        >>> REGISTRY.get("SubClass")
        <class '__main__.SubClass'>
        >>> BaseClass.REGISTRY.get("SubClass")
        <class '__main__.SubClass'>
    """

    REGISTRY: Registry

    def __new__(
        cls,
        name: str,
        bases: Tuple[type, ...],
        attrs: Dict[str, type],
        register: bool = True,
        register_name: Optional[str] = None,
        registry: Optional[Registry] = None,
    ):
        """Automatically register the class.

        Args:
            name (str): Class name
            bases (tuple): Base classes
            attrs (dict): Class attributes
            register (bool, optional): Weather to register the class. Defaults to True.
              Should be set to False for abstract base classes.
            register_name (str | None, optional): Name used for registration.
              If unset, the class name is used. Defaults to None.
            registry (Registry | None, optional): Registry to use for registration.
              Defaults to None. Has to be set in the base class.
        """
        new_class = super().__new__(cls, name, bases, attrs)
        if not hasattr(new_class, "REGISTRY"):
            if registry is not None:
                new_class.REGISTRY = registry
            elif register:
                raise ValueError(
                    "Registry has to be set in the base class or passed as an argument."
                )
        if register:
            (registry or new_class.REGISTRY).register_module(
                name=register_name or name, module=new_class
            )
        return new_class
