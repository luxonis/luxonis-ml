import warnings
from abc import ABCMeta
from typing import (
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T", bound=type)


class Registry(Generic[T]):
    def __init__(self, name: str):
        """A Registry class to store and retrieve modules.

        @type name: str
        @ivar name: Name of the registry
        """
        self._module_dict: Dict[str, T] = {}
        self._name = name

    def __str__(self) -> str:
        return f"Registry('{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self._module_dict)

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __setitem__(self, key: str, value: T) -> None:
        self._register(module=value, module_name=key, force=True)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    @property
    def name(self) -> str:
        return self._name

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
            raise KeyError(f"'{key}' not in the '{self.name}' registry.")
        return module_cls

    @overload
    def register_module(
        self, name: Optional[str] = ..., module: None = ..., force: bool = ...
    ) -> Callable[[T], T]: ...

    @overload
    def register_module(
        self, name: Optional[str] = ..., module: T = ..., force: bool = ...
    ) -> T: ...

    def register_module(
        self,
        name: Optional[str] = None,
        module: Optional[T] = None,
        force: bool = False,
    ) -> Union[T, Callable[[T], T]]:
        warnings.warn(
            "`register_module` is deprecated, use `register` instead.",
            stacklevel=2,
        )

        return self.register(name=name, module=module, force=force)

    @overload
    def register(
        self,
        module: None = ...,
        *,
        name: Optional[str] = ...,
        force: bool = ...,
    ) -> Callable[[T], T]: ...

    @overload
    def register(
        self,
        module: T = ...,
        *,
        name: Optional[str] = ...,
        force: bool = ...,
    ) -> None: ...

    def register(
        self,
        module: Optional[T] = None,
        *,
        name: Optional[str] = None,
        force: bool = False,
    ) -> Optional[Callable[[T], T]]:
        """Registers a module.

        Can be used as a decorator or as a normal method:

            >>> registry = Registry(name="modules")
            >>> @registry.register()
            ... class Foo:
            ...     pass
            >>> registry.get("Foo")
            <class '__main__.Foo'>
            >>> class Bar:
            ...     pass
            >>> registry.register(module=Bar)
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

        @rtype: Union[type, Callable[[type], type]]
        @return: Module class or register function if used as a decorator

        @raise KeyError: Raised if class name already exists and C{force==False}
        """

        if module is not None:
            return self._register(module=module, module_name=name, force=force)

        def wrapper(module: T) -> T:
            self._register(module=module, module_name=name, force=force)
            return module

        return wrapper

    def _register(
        self, module: T, module_name: Optional[str] = None, force: bool = False
    ) -> None:
        if module_name is None:
            module_name = module.__name__

        if not force and module_name in self._module_dict:
            existed_module = self._module_dict[module_name]
            raise KeyError(
                f"`{module_name}` already registred in `{self.name}` "
                f"registry at `{existed_module.__module__}`."
            )

        self._module_dict[module_name] = module


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
            registry.register(name=register_name or name, module=new_class)
        return new_class
