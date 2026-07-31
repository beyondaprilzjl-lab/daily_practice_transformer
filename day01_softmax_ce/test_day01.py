"""Automated checks for the Day 01 implementation."""

import importlib
import os
import unittest

import numpy as np


MODULE_UNDER_TEST = os.environ.get("DAY01_MODULE", "solution")
implementation = importlib.import_module(MODULE_UNDER_TEST)
cross_entropy = implementation.cross_entropy
stable_log_softmax = implementation.stable_log_softmax
stable_softmax = implementation.stable_softmax


def reference_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    normalizer = np.logaddexp.reduce(logits, axis=axis, keepdims=True)
    return logits - normalizer


class StableSoftmaxTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [-2.0, 0.0, 5.0]])
        probabilities = stable_softmax(logits)
        np.testing.assert_allclose(
            probabilities.sum(axis=-1),
            np.ones(2),
            atol=1e-12,
        )

    def test_supports_non_last_axis(self) -> None:
        logits = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        probabilities = stable_softmax(logits, axis=1)
        self.assertEqual(probabilities.shape, logits.shape)
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones((2, 4)),
            atol=1e-12,
        )

    def test_is_invariant_to_constant_shift(self) -> None:
        logits = np.array([[0.5, -1.0, 3.0]], dtype=np.float64)
        np.testing.assert_allclose(
            stable_softmax(logits),
            stable_softmax(logits + 10000.0),
            atol=1e-12,
        )

    def test_extreme_logits_are_finite(self) -> None:
        logits = np.array(
            [[10000.0, 9999.0, -10000.0], [-10000.0, -10001.0, -9999.0]]
        )
        result = stable_softmax(logits)
        self.assertTrue(np.isfinite(result).all())

    def test_equal_logits_produce_uniform_probabilities(self) -> None:
        logits = np.full((2, 4), 7.0)
        np.testing.assert_allclose(stable_softmax(logits), np.full((2, 4), 0.25))

    def test_rejects_invalid_axis_and_non_floating_logits(self) -> None:
        with self.assertRaises(ValueError):
            stable_softmax(np.ones((2, 3)), axis=2)
        with self.assertRaises(TypeError):
            stable_softmax(np.ones((2, 3), dtype=np.int64))


class StableLogSoftmaxTests(unittest.TestCase):
    def test_matches_independent_reference(self) -> None:
        logits = np.array([[2.0, -1.0, 0.5], [10000.0, 9999.0, 9998.0]])
        np.testing.assert_allclose(
            stable_log_softmax(logits),
            reference_log_softmax(logits),
            atol=1e-12,
        )

    def test_matches_softmax_in_probability_space(self) -> None:
        logits = np.array([[2.0, -1.0, 0.5], [10000.0, 9999.0, 9998.0]])
        np.testing.assert_allclose(
            np.exp(stable_log_softmax(logits)),
            stable_softmax(logits),
            atol=1e-12,
        )

    def test_extreme_logits_are_finite(self) -> None:
        logits = np.array([[10000.0, -10000.0], [-10000.0, 10000.0]])
        result = stable_log_softmax(logits)
        self.assertTrue(np.isfinite(result).all())

    def test_supports_negative_and_positive_axis(self) -> None:
        logits = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        np.testing.assert_allclose(
            stable_log_softmax(logits, axis=-2),
            stable_log_softmax(logits, axis=1),
        )


class CrossEntropyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = np.array(
            [[2.0, 1.0, 0.0], [-1.0, 3.0, 0.5]],
            dtype=np.float64,
        )
        self.targets = np.array([0, 2])

    def expected_losses(self) -> np.ndarray:
        return -reference_log_softmax(self.logits)[
            np.arange(self.targets.size),
            self.targets,
        ]

    def test_none_reduction(self) -> None:
        losses = cross_entropy(self.logits, self.targets, reduction="none")
        np.testing.assert_allclose(losses, self.expected_losses(), atol=1e-12)

    def test_mean_and_sum_reductions(self) -> None:
        expected = self.expected_losses()
        self.assertAlmostEqual(
            float(cross_entropy(self.logits, self.targets, reduction="mean")),
            float(expected.mean()),
        )
        self.assertAlmostEqual(
            float(cross_entropy(self.logits, self.targets, reduction="sum")),
            float(expected.sum()),
        )

    def test_rejects_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            cross_entropy(self.logits.reshape(1, 2, 3), self.targets)
        with self.assertRaises(ValueError):
            cross_entropy(self.logits, self.targets.reshape(2, 1))
        with self.assertRaises(ValueError):
            cross_entropy(self.logits, np.array([0]))

    def test_rejects_invalid_target_or_reduction(self) -> None:
        with self.assertRaises(ValueError):
            cross_entropy(self.logits, np.array([0, 3]))
        with self.assertRaises(ValueError):
            cross_entropy(self.logits, self.targets, reduction="median")

    def test_rejects_non_integer_targets(self) -> None:
        with self.assertRaises(TypeError):
            cross_entropy(self.logits, np.array([0.0, 2.0]))

    def test_uniform_logits_have_log_num_classes_loss(self) -> None:
        logits = np.zeros((5, 4), dtype=np.float64)
        targets = np.array([0, 1, 2, 3, 0])
        expected = np.full(5, np.log(4.0))
        np.testing.assert_allclose(
            cross_entropy(logits, targets, reduction="none"),
            expected,
            atol=1e-12,
        )

    def test_extreme_logits_produce_finite_loss(self) -> None:
        logits = np.array(
            [[10000.0, 9999.0, -10000.0], [-10000.0, 10000.0, 9999.0]]
        )
        targets = np.array([0, 2])
        losses = cross_entropy(logits, targets, reduction="none")
        self.assertTrue(np.isfinite(losses).all())

    def test_rejects_empty_batch_and_non_finite_logits(self) -> None:
        with self.assertRaises(ValueError):
            cross_entropy(
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
            )
        with self.assertRaises(ValueError):
            cross_entropy(
                np.array([[1.0, np.nan]]),
                np.array([0]),
            )


if __name__ == "__main__":
    unittest.main()
