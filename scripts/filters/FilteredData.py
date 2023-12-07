""" FilteredData Class

Defines a return class for data filtered by Refactored_NoiseFiltersR

The FilteredData class is essentially a list containing certain parameters, including:

# TODO: Formalize docstrings

"""
import dataclasses
from numpy.typing import ArrayLike

@dataclasses.dataclass(frozen=True)
class FilteredData:
    clean_data: ArrayLike
    removed_idx: ArrayLike
    repaired_idx: ArrayLike
    repaired_labels: ArrayLike
    parameters: dict
    call: str
    extra_info: dict


