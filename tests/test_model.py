"""CNNNet 多项式激活的轻量检查，不依赖数据集。"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from model import CNNNet, polynom_act


class PolynomActTest(unittest.TestCase):
    def test_quadratic(self):
        act = polynom_act()
        with torch.no_grad():
            act.alpha.fill_(2.0)
            act.beta.fill_(3.0)
            act.c.fill_(4.0)
        x = torch.tensor([0.0, 1.0, -2.0])
        y = act(x)
        self.assertAlmostEqual(y[0].item(), 4.0, places=5)
        self.assertAlmostEqual(y[1].item(), 9.0, places=5)
        self.assertAlmostEqual(y[2].item(), 6.0, places=5)


class CNNNetShapeTest(unittest.TestCase):
    def test_logits_shape(self):
        net = CNNNet(10, filters=8)
        net.eval()
        x = torch.zeros(2, 3, 32, 32)
        y = net(x)
        self.assertEqual(tuple(y.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
