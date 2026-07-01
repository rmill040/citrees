from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Register callables to create a universal object builder.

    Parameters
    ----------
    name : str
        Name of registry.

    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get registry name.

        Returns
        -------
        str
            Name of registry.

        """
        return self._name

    def keys(self) -> list[Any]:
        """Return keys in registry.

        Returns
        -------
        List[Any]
            List of keys.

        """
        return list(self._registry.keys())

    def __contains__(self, key: object) -> bool:
        """Return True if alias exists in the registry.

        Without this method, Python may fall back to the sequence protocol for
        membership checks and call ``__getitem__`` with integer indices (0, 1, ...),
        which is not what this mapping-like type intends.
        """
        if not isinstance(key, str):
            return False
        return key in self._registry

    def __iter__(self):
        """Iterate over registered aliases."""
        return iter(self._registry)

    def __len__(self) -> int:
        """Return the number of registered aliases."""
        return len(self._registry)

    def __getitem__(self, key: str) -> T:  # type: ignore
        """Get item in registry.

        Parameters
        ----------
        key : str
            Key in registry.

        Returns
        -------
        T
            Key in registry

        """
        entry = self._registry.get(key, None)
        if not entry:
            raise KeyError(f"({key}) not found in registry ({self._name})")

        return entry

    def register(self, alias: str) -> Any:
        """Register callable.

        Parameters
        ----------
        alias : str
            Alias for callable.

        Returns
        -------
        T
            Callable to be registered.

        """

        def wrapper(f: T) -> T:
            # Alias must be unique
            if alias in self._registry:
                raise KeyError(f"alias ({alias}) already exists in registry ({self._name})")

            # Add callable to registry and return
            self._registry[alias] = f
            return f

        return wrapper


# Define registries
ClassifierSelectors = Registry("ClassifierSelectors")
ClassifierSelectorTests = Registry("ClassifierSelectorTests")
RegressorSelectors = Registry("RegressorSelectors")
RegressorSelectorTests = Registry("RegressorSelectorTests")
ClassifierSplitters = Registry("ClassifierSplitters")
ClassifierSplitterTests = Registry("ClassifierSplitterTests")
RegressorSplitters = Registry("RegressorSplitters")
RegressorSplitterTests = Registry("RegressorSplitterTests")
ThresholdMethods = Registry("ThresholdMethods")
