"""Exercise shared COCO evaluation without requiring a GPU or simulator."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.utils.metrics import OKS_SIGMA, kpt_iou

from coco_eval_decode_sdk import fit_mapping
from coco_eval_metrics import pose_array, prediction_rows, score
from coco_eval_simulator import SDK_OUTPUT_BYTES, job_files, stable_take, valid_job


class CocoEvalTest(unittest.TestCase):
    def test_reject_missing_empty_image(self):
        with patch("coco_eval_metrics.load_predictions", return_value=[{"image_id": 1, "pred": []}]):
            with self.assertRaisesRegex(ValueError, "lacks 1 scored"):
                prediction_rows([Path("test.jsonl")], {"bird": ["bird"]}, [1, 2])

    def test_fixed_threshold_and_crowd(self):
        coco = {
            "categories": [{"id": 16, "name": "bird"}],
            "annotations": [
                {"image_id": 1, "category_id": 16, "bbox": [0, 0, 20, 20], "iscrowd": 0},
                {"image_id": 1, "category_id": 16, "bbox": [70, 70, 20, 20], "iscrowd": 1},
            ],
        }
        preds = {
            1: [{"label": "bird", "score": .95, "bbox": [0, 0, 20, 20]}],
            2: [{"label": "bird", "score": .90, "bbox": [0, 0, 20, 20]}],
        }
        stats = score(coco, preds, {"bird": ["bird"]})["classes"]["bird"]
        self.assertEqual(stats["gt_instances"], 1)
        self.assertAlmostEqual(stats["ap50_95"], .995)
        self.assertEqual(stats["at_conf_0.85_iou_0.50"],
                         {"tp": 1, "fp": 1, "precision": .5, "recall": 1.})

    def test_missing_pose_prediction_contributes_zero_similarity(self):
        gt = torch.tensor([[[0., 0., 2.], [2., 4., 1.]]])
        pts = [[0., 0., 1.], None] + [None] * 15
        pred = pose_array([{"keypoints": pts}], [(0, 0), (1, 1)], gt=False, is_yolo=False)
        oks = kpt_iou(gt, pred, torch.tensor([800.]), OKS_SIGMA[:2].tolist())
        self.assertAlmostEqual(oks.item(), .5, places=6)

    def test_mapping_needs_local_disambiguation(self):
        distances = np.full((8, 17, 17), .5)
        for n in range(17):
            distances[:, n, (n + 1) % 17] = .02
        distances[:, 0, 0] = .02
        mapping, report = fit_mapping(distances)
        self.assertEqual(len(mapping), 16)
        self.assertFalse(report[0]["accepted"])
        self.assertIn([16, 0], mapping)

    def test_stable_sample_not_dependent_on_input_order(self):
        self.assertEqual(stable_take(range(20), "pilot", 4),
                         stable_take(reversed(range(20)), "pilot", 4))

    def test_shard_requires_completion_stamp_and_complete_tensors(self):
        item = ("human", 7, "512x336-000000000007")
        log, first, second, stamp = job_files(item)
        data = {log: b"log", first: bytes(SDK_OUTPUT_BYTES["human"][0]),
                second: bytes(SDK_OUTPUT_BYTES["human"][1])}
        self.assertFalse(valid_job(data, item))
        data[stamp] = b'{"model":"human","image_id":7}'
        self.assertTrue(valid_job(data, item))
        data[second] = b"short"
        self.assertFalse(valid_job(data, item))


if __name__ == "__main__":
    unittest.main()
