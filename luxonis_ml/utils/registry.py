from abc import ABCMeta
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

T = TypeVar("T", bound=type)


class Registry(Generic[T]):
    def __init__(self, name: str):
        """A Registry class to store and retrieve modules.

        @type name: str
        @param name: Name of the registry
        """
        self._module_dict: Dict[str, T] = {}
        self._name = name
        self._default: Optional[T] = None

    def __str__(self):
        return f"Registry('{self.name}')"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._module_dict)

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get_default(self) -> T:
        """Retrieves the default registry record.

        @rtype: type
        @return: Default class if exists
        @raise KeyError: If default class is not in the registry
        """
        if self._default is None:
            raise KeyError(
                f"`{self.name}` registry does not have a default object."
            )
        return self._default

    def get(self, key: str) -> T:
        """Retrieves the registry record for the key.

        @type key: str
        @param key: Name of the registered item, I{e.g.} the class name
            in string format.
        @rtype: type
        @return: Corresponding class if L{key} exists
        @raise KeyError: If L{key} is not in the registry
        """
        module_cls = self._module_dict.get(key, None)
        if module_cls is None:
            raise KeyError(f"Class `{key}` not in the `{self.name}` registry.")
        else:
            return module_cls

    def register_module(
        self,
        name: Optional[str] = None,
        module: Optional[T] = None,
        force: bool = False,
        mark_default: bool = False,
    ) -> Union[T, Callable[[T], T]]:
        """Registers a module.

        Can be used as a decorator or as a normal method:

            >>> registry = Registry(name="modules")
            >>> @registry.register_module()
            ... class Foo:
            ...     pass
            >>> registry.get("Foo")
            <class '__main__.Foo'>
            >>> class Bar:
            ...     pass
            >>> registry.register_module(module=Bar)
            >>> registry.get("Bar")
            <class '__main__.Bar'>

        @type name: Optional[str]
        @param name: Name of the module. If C{None}, then use class name.
            Defaults to None.

        @type module: Optional[type]
        @param module: Module class to be registered. Defaults to None.

        @type force: bool
        @param force: Whether to override an existing class with the same name.
            Defaults to False.

        @type mark_default: bool
        @param mark_default: Whether to mark the module as default.
            Default module can be retrieved with L{get_default} method.

        @rtype: Union[type, Callable[[type], type]]
        @return: Module class or register function if used as a decorator

        @raise KeyError: Raised if class name already exists and C{force==False}
        """

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module=module,
                module_name=name,
                force=force,
                mark_default=mark_default,
            )
            return module

        # use it as a decorator: @x.register_module()
        def _register(module: T) -> T:
            self._register_module(
                module=module,
                module_name=name,
                force=force,
                mark_default=mark_default,
            )
            return module

        return _register

    def _register_module(
        self,
        module: T,
        module_name: Optional[str] = None,
        force: bool = False,
        mark_default: bool = False,
    ) -> None:
        if module_name is None:
            module_name = module.__name__

        if not force and module_name in self._module_dict:
            existed_module = self._module_dict[module_name]
            raise KeyError(
                f"`{module_name}` already registred in `{self.name}` registry at `{existed_module.__module__}`."
            )

        self._module_dict[module_name] = module
        if mark_default:
            self._default = module


class AutoRegisterMeta(ABCMeta):
    """Metaclass for automatically registering modules.

    Can be set as a metaclass for abstract base classes. Then, all subclasses will be
    automatically registered under the name of the subclass.

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
        mark_default: bool = False,
    ):
        """Automatically register the class.

        @type name: str
        @param name: Class name

        @type bases: Tuple[type, ...]
        @param bases: Base classes

        @type attrs: Dict[str, type]
        @param attrs: Class attributes

        @type register: bool
        @param register: Weather to register the class. Defaults to True.
            Should be set to False for abstract base classes.

        @type register_name: Optional[str]
        @param register_name: Name used for registration.
            If unset, the class name is used. Defaults to None.

        @type registry: Optional[Registry]
        @param registry: Registry to use for registration.
            Defaults to None. Has to be set in the base class.

        @type mark_default: bool
        @param mark_default: Whether to mark the module as default.
            Default module can be retrieved with L{get_default} method.
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
            registry = registry if registry is not None else new_class.REGISTRY
            registry.register_module(
                name=register_name or name,
                module=new_class,
                mark_default=mark_default,
            )
        return new_class
