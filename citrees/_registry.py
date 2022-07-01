# from typing import Any, List, TypeVar

# T = TypeVar("T")


# class Registry:
#     """Used to register classes so that a universal object builder can be enabled.

#     Parameters
#     ----------
#     name : str
#         Name of registry.
#     """
#     def __init__(self, name: str) -> None:
#         self._name = name
#         self._registry = dict()

#     @property
#     def name(self) -> str:
#         """Get registry name.

#         Parameters
#         ----------
#         None

#         Returns
#         -------
#         str
#             Name of registry.
#         """
#         return self._name

#     def keys(self) -> List[Any]:
#         """Return keys in registry.

#         Parameters
#         ----------
#         None

#         Returns
#         -------
#         list
#             List of keys.
#         """
#         return list(self._registry.keys())

#     def __getitem__(self, key: str) -> T:
#         """Get item in registry.

#         Parameters
#         ----------
#         key : str
#             Key in registry.

#         Returns
#         -------
#         T
#             Key in registry
#         """
#         entry = self._registry.get(key, None)
#         if not entry:
#             raise KeyError(f"{key} not found in {self._name} registry")

#         return entry

#     def register_class(self, alias: str) -> Any:
#         """Register class.

#         Parameters
#         ----------
#         alias : str
#             Alias for class.

#         Returns
#         -------
#         T
#             Class to be registered.
#         """
#         def wrapper(cls: T) -> T:
#             # Alias must be unique
#             if alias in self._registry:
#                 raise KeyError(f"alias ({alias}) already exists in {self._name} registry")

#             # Add class to registry and return
#             self._registry[alias] = cls
#             return cls

#         return wrapper


# Splitters = Registry("Splitters")
# Selectors = Registry("Selectors")
