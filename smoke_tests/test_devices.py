"""GPU 路由用 mock 检查；真实多进程训练另见 run_parallel_smoke.py。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

from config import CONFIG
from evaluate import resolve_device
from train import experiment_jobs, partition_jobs, training_devices


class DeviceTests(unittest.TestCase):
    def test_cpu_fallback_and_explicit_cuda_failure(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_device("auto")), "cpu")
            self.assertEqual([str(d) for d in training_devices(replace(CONFIG, device="auto"))], ["cpu"])
            self.assertEqual(str(resolve_device("cpu")), "cpu")
            for name in ("cuda", "cuda:0"):
                with self.assertRaises(RuntimeError):
                    training_devices(replace(CONFIG, device=name))

    def test_two_gpus_and_explicit_single_gpu(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            for name in ("auto", "cuda"):
                devices = training_devices(replace(CONFIG, device=name, gpu_ids=(0, 1)))
                self.assertEqual([str(d) for d in devices], ["cuda:0", "cuda:1"])
            self.assertEqual([str(d) for d in training_devices(replace(CONFIG, device="cuda:1"))], ["cuda:1"])
            self.assertEqual([str(d) for d in training_devices(replace(CONFIG, device="cuda", gpu_ids=(1,)))], ["cuda:1"])
            self.assertEqual(str(resolve_device("cuda")), "cuda:0")
            with self.assertRaises(ValueError):
                resolve_device("cuda:2")

    def test_auto_partial_pool_and_strict_cuda_pool(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            self.assertEqual([str(d) for d in training_devices(replace(CONFIG, device="auto"))], ["cuda:0"])
            with self.assertRaises(ValueError):
                training_devices(replace(CONFIG, device="cuda", gpu_ids=(0, 1)))
            with self.assertRaises(ValueError):
                training_devices(replace(CONFIG, device="auto", gpu_ids=(1,)))

    def test_invalid_device_configuration(self):
        for key in ("device", "evaluation_device"):
            for value in ("gpu", "cuda:-1", "cuda:", 0):
                with self.assertRaises(ValueError):
                    replace(CONFIG, **{key: value}).validate()
        for ids in ((), (0, 0), (-1,), (True,), (0.0,)):
            with self.assertRaises(ValueError):
                replace(CONFIG, gpu_ids=ids).validate()

    def test_jobs_cover_all_folds_and_seeds_once(self):
        cfg = replace(CONFIG, N=3, M=3, seed=42, windows_ms=(500, 1000))
        jobs = experiment_jobs(cfg)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(set(jobs), {(w, s, r) for w in (500, 1000) for s in (1, 2, 3) for r in (42, 43, 44)})
        for workers in (1, 2):
            parts = [partition_jobs(jobs, rank, workers) for rank in range(workers)]
            flattened = [job for part in parts for job in part]
            self.assertCountEqual(flattened, jobs)
            self.assertEqual(len(set(flattened)), len(jobs))
        with self.assertRaises(ValueError):
            partition_jobs(jobs, 2, 2)


if __name__ == "__main__":
    unittest.main()
