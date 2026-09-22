"""CPU regression test of DDIM time ordering, separate from learned inversion quality."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'dp_adapter'))
from libero_fm import integrate


class ZeroEpsilon:
    def field(self,x,t,c):return torch.zeros_like(x)


class DDIMTests(unittest.TestCase):
    def test_zero_epsilon_roundtrip_without_clipping(self):
        action=torch.linspace(-.2,.2,112).reshape(1,16,7)
        c=torch.zeros(1,1)
        for n in (10,100):
            source=integrate(ZeroEpsilon(),c,action,n,-1,'ddim')
            recovered=integrate(ZeroEpsilon(),c,source,n,1,'ddim')
            torch.testing.assert_close(recovered,action,rtol=1e-4,atol=1e-6)

    def test_fm_solver_rejected(self):
        with self.assertRaises(ValueError):
            integrate(ZeroEpsilon(),torch.zeros(1,1),torch.zeros(1,16,7),10,solver='rk4')


if __name__=='__main__':unittest.main()
