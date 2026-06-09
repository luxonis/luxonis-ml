import warnings
from abc import ABCMeta
from collections.abc import Callable
from typing import Generic, TypeVar, overload

from typing_extensions import deprecated

T = TypeVar("T", bound=type)


class Registry(Generic[T]):
    """A registry to store and retrieve modules.

    The registry stores a mapping from string keys to module classes. It
    provides an interface to register modules and later retrieve them by
    their string names.
    """

    def __init__(self, name: str):
        """Create a new empty registry.

        Args:
            name: The name of the registry, used for error messages.
        """
        self._module_dict: dict[str, T] = {}
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
        """The name of the registry."""
        return self._name

    def get(self, key: str) -> T:
        """Retrieves the registry record for the key.

        Args:
            key: The name of the registered item.
        Returns:
            The registered item corresponding to the key.
        Raises:
            KeyError: If the key is not found in the registry.
        """
        module_cls = self._module_dict.get(key, None)
        if module_cls is None:
            raise KeyError(f"'{key}' not in the '{self.name}' registry.")
        return module_cls

    @overload
    def register_module(
        self, name: str | None = ..., module: None = ..., force: bool = ...
    ) -> Callable[[T], T]: ...

    @overload
    def register_module(
        self, name: str | None = ..., module: T = ..., force: bool = ...
    ) -> T: ...

    @deprecated("`register_module` is deprecated, use `register` instead.")
    def register_module(
        self,
        name: str | None = None,
        module: T | None = None,
        force: bool = False,
    ) -> T | Callable[[T], T]:
        """Registers a module. Deprecated, use `register` instead.

        Args:
            name: Name of the module. If ``None``, the class name is used.
            module: Module class to be registered.
            force: Whether to override an existing class with the same name.
        """
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
        name: str | None = ...,
        force: bool = ...,
    ) -> Callable[[T], T]: ...

    @overload
    def register(
        self,
        module: T = ...,
        *,
        name: str | None = ...,
        force: bool = ...,
    ) -> None: ...

    def register(
        self,
        module: T | None = None,
        *,
        name: str | None = None,
        force: bool = False,
    ) -> Callable[[T], T] | None:
        """Registers a module.

        Can be used as a decorator or as a normal method:

        Example:

            >>> registry = Registry(name="modules")
            >>> @registry.register()
            ... class Foo:
            ...     pass
            >>> registry.get("Foo").__name__
            'Foo'
            >>> class Bar:
            ...     pass
            >>> registry.register(module=Bar)
            >>> registry.get("Bar").__name__
            'Bar'

        Args:
            name: The name of the module. If ``None``, then use class name.
            module: The module class to be registered.
            force: Whether to override an existing class with the same name.

        Returns:
            ``None`` if used as a normal method,
            or a decorator function if used as a decorator.

        Raises:
            KeyError: If a module with the same name
                already exists and ``force`` is ``False``.
        """

        if module is not None:
            return self._register(module=module, module_name=name, force=force)

        def wrapper(module: T) -> T:
            self._register(module=module, module_name=name, force=force)
            return module

        return wrapper

    def _register(
        self, module: T, module_name: str | None = None, force: bool = False
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

    Can be set as a metaclass for abstract base classes.
    All subclasses of that class will be automatically registered.

    Attributes:
        REGISTRY: The internal registry defined on the base class.

    Example:

        >>> REGISTRY = Registry(name="modules")
        >>> class Base(
        ...     metaclass=AutoRegisterMeta,
        ...     registry=REGISTRY,
        ...     register=False
        ... ):
        ...     pass
        >>> class Foo(Base):
        ...     pass
        >>> class Bar(Base, register_name="Baz"):
        ...     pass
        >>> REGISTRY.get("Foo").__name__
        'Foo'
        >>> Base.REGISTRY.get("Baz").__name__
        'Bar'
    """

    REGISTRY: Registry

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, type],
        register: bool = True,
        register_name: str | None = None,
        registry: Registry | None = None,
    ):
        """Automatically register the class.

        Args:
            name: The name of the class being created.
            bases: The base classes of the class being created.
            attrs: The attributes of the class being created.
            register: Whether to register this class.
                Typically should be set to ``False`` for abstract base classes.
            register_name: The name to register the class under. If ``None``, then use
                the class name.
            registry: The registry to use for registration.
                Will be saved as the `REGISTRY` attribute
                of the class if not already set.
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
            registry.register(
                name=register_name or name, module=new_class, force=True
            )
        return new_class
