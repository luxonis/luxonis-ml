import types

import pytest

from luxonis_ml.utils.registry import AutoRegisterMeta, Registry


@pytest.fixture
def registry() -> Registry[type]:
    return Registry("test")


def test_creation():
    registry = Registry("test")
    assert registry.name == "test"
    assert not registry._module_dict
    assert len(registry) == 0
    assert str(registry) == "Registry('test')"
    assert repr(registry) == "Registry('test')"


def test_registry(registry: Registry[type]):
    assert not registry._module_dict
    assert len(registry) == 0

    class A:
        pass

    registry.register(module=A)
    assert registry.get("A") is A

    @registry.register()
    class B:
        pass

    assert registry.get("B") is B
    assert len(registry) == 2

    registry.register(name="C", module=A)
    assert registry.get("C") is A
    assert len(registry) == 3

    registry.register(name="C", module=B, force=True)
    assert registry.get("C") is B

    registry["D"] = A
    assert "D" in registry
    assert registry["D"] is A

    with pytest.raises(KeyError):
        registry.register(name="C", module=A, force=False)

    @registry.register(name="Foo")
    class Bar:
        pass

    assert registry.get("Foo") is Bar
    with pytest.raises(KeyError):
        registry.get("Bar")


def test_autoregistry(registry: Registry):
    assert len(registry) == 0

    class Base(metaclass=AutoRegisterMeta, registry=registry, register=False):
        pass

    assert len(registry) == 0

    class A(Base):
        pass

    class B(Base):
        pass

    assert len(registry) == 2
    assert registry.get("A") is A
    assert registry.get("B") is B

    class C(Base, register_name="AliasC"):
        pass

    assert len(registry) == 3
    assert registry.get("AliasC") is C
    with pytest.raises(KeyError):
        registry.get("C")

    class D(Base, register=False):
        pass

    assert len(registry) == 3
    with pytest.raises(KeyError):
        registry.get("D")

    assert D.REGISTRY is registry

    foo_registry = Registry("foo")

    class E(Base, registry=foo_registry):
        pass

    assert len(registry) == 3
    with pytest.raises(KeyError):
        registry.get("E")

    assert len(foo_registry) == 1
    assert foo_registry.get("E") is E

    with pytest.raises(ValueError, match="Registry has to be set"):

        class _(metaclass=AutoRegisterMeta, register=True):
            pass


def test_autoregistry_decorator(registry: Registry[type]):
    base = registry.autoregister()(types.new_class("Base"))
    a = types.new_class("A", (base,))
    b = types.new_class("B", (base,), {"register_name": "AliasB"})
    c = types.new_class("C", (base,), {"register": False})

    assert registry.get("A") is a
    assert registry.get("AliasB") is b
    assert "B" not in registry
    assert c.__name__ == "C"
    assert "C" not in registry


def test_autoregistry_decorator_keeps_existing_hooks(
    registry: Registry[type],
):
    calls: list[tuple[str, str]] = []
    other = Registry[type]("other")

    class Parent:
        def __init_subclass__(cls) -> None:
            calls.append(("parent", cls.__name__))
            super().__init_subclass__()

    class Base(Parent):
        def __init_subclass__(cls, *, tag: str) -> None:
            assert cls.__name__ not in registry
            assert cls.__name__ not in other
            calls.append((tag, cls.__name__))
            super().__init_subclass__()

    calls.clear()
    base = other.autoregister()(registry.autoregister()(Base))
    child = types.new_class("Child", (base,), {"tag": "base"})

    assert calls == [("base", "Child"), ("parent", "Child")]
    assert registry.get("Child") is child
    assert other.get("Child") is child

    alias = types.new_class(
        "Aliased", (base,), {"tag": "base", "register_name": "Alias"}
    )
    skipped = types.new_class(
        "Skipped", (base,), {"tag": "base", "register": False}
    )
    for registered in (registry, other):
        assert registered.get("Alias") is alias
        assert "Aliased" not in registered
        assert skipped.__name__ not in registered


def test_autoregistry_decorator_forwards_inherited_controls(
    registry: Registry[type],
):
    other = Registry[type]("other")
    ancestor = registry.autoregister()(types.new_class("Ancestor"))
    base = other.autoregister()(types.new_class("Base", (ancestor,)))

    alias = types.new_class("Aliased", (base,), {"register_name": "Alias"})
    skipped = types.new_class("Skipped", (base,), {"register": False})

    for registered in (registry, other):
        assert registered.get("Alias") is alias
        assert "Aliased" not in registered
        assert skipped.__name__ not in registered


def test_autoregistry_decorator_handles_same_named_classes(
    registry: Registry[type],
):
    base = registry.autoregister()(types.new_class("Base"))
    first = type("Node", (base,), {})

    class Proxy:
        pass

    proxy = type("Node", (Proxy, first), {})
    assert issubclass(proxy, first)
    assert registry.get("Node") is first

    second = type("Node", (base,), {})
    assert first is not second
    assert registry.get("Node") is second


def test_autoregistry_keeps_class_over_same_named_subclass(
    registry: Registry,
):
    class Base(metaclass=AutoRegisterMeta, registry=registry, register=False):
        pass

    class Node(Base):
        pass

    # The tracer of `torch.export` wraps a module in a subclass of the
    # module class, under the same name.
    class Proxy:
        pass

    proxy = type("Node", (Proxy, Node), {})

    assert issubclass(proxy, Node)
    assert registry.get("Node") is Node


def test_autoregistry_replaces_redefined_class(registry: Registry):
    class Base(metaclass=AutoRegisterMeta, registry=registry, register=False):
        pass

    # `importlib.reload` defines the class again, without inheritance.
    first = type("Node", (Base,), {})
    second = type("Node", (Base,), {})

    assert first is not second
    assert registry.get("Node") is second


def test_autoregistry_rejects_unknown_class_keywords(registry: Registry):
    class Base(metaclass=AutoRegisterMeta, registry=registry, register=False):
        pass

    with pytest.raises(TypeError, match="alias"):
        types.new_class("Node", (Base,), {"alias": "Other"})
    assert len(registry) == 0
