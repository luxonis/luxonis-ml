import pytest

from luxonis_ml.utils.registry import AutoRegisterMeta, Registry


@pytest.fixture
def registry() -> Registry:
    return Registry("test")


def test_creation():
    registry = Registry("test")
    assert registry.name == "test"
    assert not registry._module_dict
    assert len(registry) == 0
    assert str(registry) == "Registry('test')"
    assert repr(registry) == "Registry('test')"


def test_registry(registry: Registry):
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
