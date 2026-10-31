"""Checkpoint regressions runnable without optional ML dependencies."""

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SRC = Path(__file__).resolve().parents[1] / "src" / "fastdrs"


class CheckpointRegressionTests(unittest.TestCase):
    def test_preprocessing_roundtrip_preserves_image_size(self):
        config_type = runpy.run_path(str(SRC / "checkpoints/schema.py"))["PreprocessingConfig"]
        config = config_type(img_size=384, ben_graham=True)
        self.assertEqual(config_type.from_dict(config.to_dict()), config)
        self.assertEqual(config_type.from_dict({}).img_size, 224)

    def test_self_defining_export_uses_checkpoint_size(self):
        model = Mock()
        loaded = SimpleNamespace(model=model, preprocessing=SimpleNamespace(img_size=384))
        loader = Mock(return_value=loaded)
        torch = SimpleNamespace(randn=Mock(return_value="sample"))
        edge = Mock()
        converter = Mock(return_value=edge)
        modules = {
            "torch": torch,
            "litert_torch": SimpleNamespace(convert=converter),
            "fastdrs": SimpleNamespace(),
            "fastdrs.checkpoints": SimpleNamespace(load_checkpoint=loader),
            "fastdrs.models": SimpleNamespace(build_model=Mock()),
        }
        with patch.dict(sys.modules, modules):
            export = runpy.run_path(str(SRC / "export/litert.py"))["export_litert"]
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                checkpoint = Path(directory) / "model.fastdrs"
                checkpoint.touch()
                output = Path(directory) / "model.tflite"
                edge.export.side_effect = lambda path: Path(path).touch()
                self.assertEqual(export(checkpoint, output), str(output))
                loader.assert_called_once_with(checkpoint, device="cpu")
                torch.randn.assert_called_once_with(1, 3, 384, 384)
                converter.assert_called_once_with(model, ("sample",))


if __name__ == "__main__":
    unittest.main()
