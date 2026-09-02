"""Unit tests for classifier models across model complexity spectrum."""

import unittest
import torch
import numpy as np
from src.models.logistic_regression import MultiClassLogisticRegression
from src.models.mlp import TwoLayerMLP
from src.models.resnet import PreActResNet18
from src.models.decision_tree import DecisionTreeModel, RandomForestModel


class TestModels(unittest.TestCase):
    def test_logistic_regression(self):
        batch_size = 8
        in_features = 32 * 32 * 3
        num_classes = 10
        
        model = MultiClassLogisticRegression(in_features=in_features, num_classes=num_classes)
        x = torch.randn(batch_size, 3, 32, 32)
        logits = model(x)
        
        self.assertEqual(logits.shape, (batch_size, num_classes))

    def test_mlp(self):
        batch_size = 4
        in_features = 784
        num_classes = 10
        
        model = TwoLayerMLP(in_features=in_features, hidden_dim=128, num_classes=num_classes)
        x = torch.randn(batch_size, 1, 28, 28)
        logits = model(x)
        
        self.assertEqual(logits.shape, (batch_size, num_classes))

    def test_preact_resnet18(self):
        batch_size = 2
        num_classes = 10
        
        model = PreActResNet18(num_classes=num_classes, in_channels=3)
        x = torch.randn(batch_size, 3, 32, 32)
        logits, features = model(x, return_features=True)
        
        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(features.shape, (batch_size, 512))

    def test_decision_tree_and_random_forest(self):
        num_samples = 100
        num_features = 10
        num_classes = 3
        
        X = np.random.randn(num_samples, num_features)
        y = np.random.randint(0, num_classes, size=num_samples)
        
        dt = DecisionTreeModel(max_depth=5)
        dt.fit(X, y)
        probs_dt = dt.predict_proba(X)
        self.assertEqual(probs_dt.shape, (num_samples, num_classes))
        self.assertTrue(np.allclose(probs_dt.sum(axis=1), 1.0))
        
        rf = RandomForestModel(n_estimators=10, max_depth=5)
        rf.fit(X, y)
        probs_rf = rf.predict_proba(X)
        self.assertEqual(probs_rf.shape, (num_samples, num_classes))
        self.assertTrue(np.allclose(probs_rf.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
