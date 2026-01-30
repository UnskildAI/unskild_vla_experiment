"""Model, dataset, loss, trainer registries for pluggable components."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class _Registry(dict[str, type[Any]]):
    """Generic registry: name -> class or factory."""

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            key = name if name is not None else getattr(cls, "__name__", cls.__qualname__)
            self[key] = cls
            return cls

        return decorator

    def get(self, name: str, **kwargs: Any) -> Any:
        cls = self[name]
        return cls(**kwargs)

    def build(self, name: str, config: Any, **kwargs: Any) -> Any:
        cls = self[name]
        return cls(config=config, **kwargs)


DatasetRegistry: _Registry = _Registry()
ModelRegistry: _Registry = _Registry()
LossRegistry: _Registry = _Registry()
TrainerRegistry: _Registry = _Registry()
