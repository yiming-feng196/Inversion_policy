"""CPU regression checks for splits, normalization, alignment and ODE direction."""
import os
import unittest
import numpy as np
import torch
from libero_fm import Policy, error_summary, integrate, norm_params, normalize, split_episodes, window_rows
from run_queue import gpu_is_idle


class ProtocolTests(unittest.TestCase):
    def test_gpu_resource_guard(self):
        idle = {"compute_pids":[], "memory_used_mib":131, "utilization_percent":0}
        self.assertTrue(gpu_is_idle(idle))
        self.assertFalse(gpu_is_idle({**idle, "compute_pids":[123]}))
        self.assertFalse(gpu_is_idle({**idle, "memory_used_mib":24000}))
        self.assertFalse(gpu_is_idle({**idle, "utilization_percent":100}))

    def test_episode_level_error_summary(self):
        values = np.asarray([1., 3., 8.]).reshape(3,1,1)
        report = error_summary(values, np.zeros_like(values), ["a","a","b"])
        self.assertEqual(report["mean_per_sample_rmse"],4.)
        self.assertEqual(report["episode_balanced_mean"],5.)
        self.assertEqual(report["per_episode_mean_rmse"],{"a":2.,"b":8.})
        self.assertEqual(report["max_per_sample_rmse"],8.)

    def test_split_disjoint_and_stable(self):
        names = [f"demo_{i}" for i in range(50)]
        split = split_episodes(names)
        self.assertEqual([len(split[s]) for s in ("train","val","test")],[35,5,10])
        self.assertEqual(len(set(sum(split.values(),[]))),50)
        self.assertEqual(split,split_episodes(names))

    def test_window_alignment_no_padding(self):
        rows = window_rows(100)
        self.assertEqual(rows[0],(1,0))
        self.assertEqual(rows[-1],(85,84))
        for cur,start in rows:
            self.assertEqual(start+1,cur)
            self.assertGreaterEqual(cur-1,0)
            self.assertLessEqual(start+16,100)

    def test_training_only_normalizer_and_no_clipping(self):
        train = np.array([[0.,4.],[2.,4.]])
        params = norm_params([train])
        np.testing.assert_allclose(normalize(train,params),[[-1,0],[1,0]])
        self.assertGreater(normalize(np.array([[100.,4.]]),params)[0,0],1.)

    def test_ode_forward_inverse(self):
        class Field:
            def field(self,x,t,c): return .3*x+c[:,None,:]
        torch.manual_seed(3)
        x = torch.randn(2,16,7)
        c = torch.randn(2,7)
        y = integrate(Field(),c,x,32)
        recovered = integrate(Field(),c,y,32,-1)
        torch.testing.assert_close(x,recovered,rtol=1e-5,atol=1e-5)
        with self.assertRaises(ValueError):
            integrate(Field(),c,x,0)
        with self.assertRaises(ValueError):
            integrate(Field(),c,x,4,solver="invalid")

    def test_author_backbones_shapes_gradients_and_eval(self):
        repo = os.environ["Q1_AUTHOR_REPO"]
        torch.set_num_threads(2)
        vision = None
        for architecture in ("unet","dit"):
            model = Policy(repo,architecture,seed=5)
            current = {k:v.detach().clone() for k,v in model.vision.state_dict().items()}
            if vision is not None:
                for key in vision:
                    torch.testing.assert_close(current[key],vision[key],rtol=0,atol=0)
            vision=current
            x = torch.randn(2,16,7,requires_grad=True)
            c = torch.randn(2,2064)
            t = torch.tensor([.2,.7])
            out = model.field(x,t,c)
            self.assertEqual(out.shape,x.shape)
            out.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            model.eval()
            torch.testing.assert_close(model.field(x,t,c),model.field(x,t,c),rtol=0,atol=0)
            del model,current,out,x

if __name__=="__main__":
    unittest.main(verbosity=2)
