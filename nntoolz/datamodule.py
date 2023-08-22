from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn
from typing_extensions import dataclass_transform


@dataclass_transform()
def _datamodule(
    cls=None,
    /,
    *,
    kw_only: bool = False,
    _ensure_inheritance: bool = True,
):
    def wrap(cls):
        old_new = cls.__new__

        def new_new(cls, *args, **kwargs):  # pylint: disable=unused-argument
            if _ensure_inheritance and not issubclass(cls, nn.Module):
                raise ValueError("Should inherit!")

            instance = old_new(cls)
            super(cls, instance).__init__()
            return instance

        cls.__new__ = new_new
        wrapper = dataclass(
            init=True,
            repr=False,
            eq=False,
            order=False,
            unsafe_hash=True,
            frozen=False,
            match_args=True,
            kw_only=kw_only,
            slots=False,
        )(cls)

        return wrapper

    # See if we're being called as @dataclass or @dataclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @dataclass without parens.
    return wrap(cls)


if TYPE_CHECKING:
    datamodule = dataclass
else:
    datamodule = _datamodule
