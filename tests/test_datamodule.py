from typing import Optional

import pytest as pt
import torch
from torch import nn
from torch._dynamo.exc import BackendCompilerFailed
from torch.fx import symbolic_trace  # type: ignore

from nntoolz.datamodule import datamodule


@datamodule
class MyModule(nn.Module):
    a: int
    b: nn.Linear

    def forward(self, x):
        return self.b(x) + self.a


@pt.fixture(scope="function")
def mymodule():
    return MyModule(a=1, b=nn.Linear(1, 1))


def test_datamodule_base(mymodule: MyModule):
    assert mymodule.a == 1
    assert isinstance(mymodule.b, nn.Linear)

    x = torch.zeros(1, 1)
    with torch.no_grad():
        y = mymodule.eval()(x)

    assert y.shape == x.shape


@pt.mark.skip(reason="No support for torch.jit.script")
def test_datamodule_jit_script(mymodule: MyModule):
    x = torch.zeros(1, 1)

    scripted = torch.jit.script(mymodule).eval()  # type: ignore
    with torch.no_grad():
        y = scripted(x)

    assert y.shape == x.shape


def test_datamodule_jit_trace(mymodule: MyModule):
    x = torch.zeros(1, 1)

    traced = torch.jit.trace(mymodule, x).eval()  # type: ignore
    with torch.no_grad():
        y = traced(x)

    assert y.shape == x.shape


def test_datamodule_compile(mymodule: MyModule):
    x = torch.zeros(1, 1)
    compiled = torch.compile(mymodule).eval()
    with torch.no_grad():
        try:
            y = compiled(x)
        except BackendCompilerFailed as e:
            pt.skip(f"{e}")

    assert y.shape == x.shape


def test_datamodule_fx(mymodule: MyModule):
    symbolic_trace(mymodule)


def test_datamodule_default_values():
    @datamodule
    class ModuleWithDefaults(nn.Module):
        a: Optional[int] = None
        b: Optional[nn.Module] = None

    mymodule = ModuleWithDefaults(a=1, b=nn.Linear(1, 1))
    assert mymodule.a == 1
    assert isinstance(mymodule.b, nn.Linear)
