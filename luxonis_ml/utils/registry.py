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

from typing_extensions import deprecated

T = TypeVar("T", bound=type)


class Registry(Generic[T]):
    def __init__(self, name: str):
        """A Registry class to store and retrieve modules.

        @type name: str
        @ivar name: Name of the registry
        """
        self._module_dict: Dict[str, T] = {}
        self._name = name

    def __str__(self):
        return f"Registry('{self.name}')"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._module_dict)

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __setitem__(self, key: str, value: T) -> None:
        self._register(module=value, module_name=key, force=True)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    @property
    def name(self):
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
        else:
            return module_cls

    @overload
    def register_module(
        self, name: Optional[str] = ..., module: None = ..., force: bool = ...
    ) -> Callable[[T], T]: ...

    @overload
    def register_module(
        self, name: Optional[str] = ..., module: T = ..., force: bool = ...
    ) -> T: ...

    @deprecated(
        "Method `register_module` is deprecated, use `register` instead.",
        category=DeprecationWarning,
    )
    def register_module(
        self,
        name: Optional[str] = None,
        module: Optional[T] = None,
        force: bool = False,
    ) -> Union[T, Callable[[T], T]]:
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

    def autoregister(self) -> Callable[[T], T]:
        """Decorator that can be used to automatically register
        subclasses of the decorated class.

        Can be used instead of L{AutoRegisterMeta} metaclass, this
        can be useful when multiple custom metaclasses are used.

        Example:

            >>> REGISTRY = Registry(name="modules")
            >>> @REGISTRY.autoregister()
            ... class Foo:
            ...     pass
            >>> class Bar(Foo):
            ...     pass
            >>> REGISTRY.get("Bar")
            <class '__main__.Bar'>
            >>> class Baz(Foo, register=False):
            ...     pass
            >>> print("Baz" in REGISTRY)
            >>> False


        @warning: Due to the limitations of Python's typing system,
        passing arguments `register` or `register_name` will be
        understood by type checkers as an error. This is a false
        positive and can be safely ignored.
        """

        def wrapper(cls_: T) -> T:
            @classmethod
            def __init_subclass__(
                cls: T,
                register: bool = True,
                register_name: Optional[str] = None,
                **kwargs,
            ) -> None:
                name = register_name or cls.__name__
                if register:
                    self.register(module=cls, name=name)
                super(cls).__init_subclass__(**kwargs)

            cls_.__init_subclass__ = __init_subclass__
            return cls_

        return wrapper


class AutoRegisterMeta(ABCMeta):
    """Metaclass for automatically registering modules.

    Can be set as a metaclass for abstract base classes. Then, all subclasses will be
    automatically registered under the name of the subclass.

    Example:

        >>> REGISTRY = Registry(name="modules")
        >>> class Foo(metaclass=AutoRegisterMeta, registry=REGISTRY):
        ...     pass
        >>> class Bar(Foo):
        ...     pass
        >>> REGISTRY.get("Bar")
        <class '__main__.Bar'>
        >>> class Baz(Foo, register=False):
        ...     pass
        >>> print("Baz" in REGISTRY)
        >>> False
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
