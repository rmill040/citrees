from typing import Any, Dict, List, TypeVar

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
        self._registry: Dict[str, Any] = dict()

    @property
    def name(self) -> str:
        """Get registry name.

        Returns
        -------
        str
            Name of registry.
        """
        return self._name

    def keys(self) -> List[Any]:
        """Return keys in registry.

        Returns
        -------
        List[Any]
            List of keys.
        """
        return list(self._registry.keys())

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
