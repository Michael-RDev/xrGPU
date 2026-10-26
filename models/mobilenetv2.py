"""MobileNetV2 1.0 architecture compatible with TorchVision's pinned V1 weights.

The layer structure follows TorchVision v0.24.0's BSD-3-Clause implementation.
Only PyTorch is needed; no TorchVision native extension is required for export.
"""

from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "torchvision/mobilenet_v2"
WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
WEIGHTS_SHA256 = "b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889"
REVISION = "IMAGENET1K_V1:" + WEIGHTS_SHA256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def architecture():
    import torch
    from torch import nn

    class ConvNormActivation(nn.Sequential):
        def __init__(self, incoming, outgoing, kernel=3, stride=1, groups=1):
            super().__init__(nn.Conv2d(incoming, outgoing, kernel, stride, (kernel - 1) // 2,
                                      groups=groups, bias=False),
                             nn.BatchNorm2d(outgoing), nn.ReLU6(inplace=False))

    class InvertedResidual(nn.Module):
        def __init__(self, incoming, outgoing, stride, expansion):
            super().__init__()
            hidden = incoming * expansion
            self.residual = stride == 1 and incoming == outgoing
            layers = []
            if expansion != 1:
                layers.append(ConvNormActivation(incoming, hidden, kernel=1))
            layers.extend([ConvNormActivation(hidden, hidden, stride=stride, groups=hidden),
                           nn.Conv2d(hidden, outgoing, 1, bias=False), nn.BatchNorm2d(outgoing)])
            self.conv = nn.Sequential(*layers)

        def forward(self, value):
            output = self.conv(value)
            return value + output if self.residual else output

    class MobileNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [ConvNormActivation(3, 32, stride=2)]
            incoming = 32
            for expansion, outgoing, repeats, stride in (
                (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
                (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
            ):
                for i in range(repeats):
                    layers.append(InvertedResidual(incoming, outgoing, stride if i == 0 else 1, expansion))
                    incoming = outgoing
            layers.append(ConvNormActivation(incoming, 1280, kernel=1))
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1000))

        def forward(self, value):
            value = self.features(value)
            value = torch.nn.functional.adaptive_avg_pool2d(value, (1, 1))
            return self.classifier(torch.flatten(value, 1))

    return MobileNetV2().eval()


def load(offline=False):
    import torch
    from runtime.model_bundle import sha256

    weights = ROOT / "artifacts/cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"
    if not weights.exists():
        if offline:
            raise FileNotFoundError("Pinned MobileNetV2 weights are not cached: " + str(weights))
        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_suffix(".download")
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, temporary)
            if sha256(temporary) != WEIGHTS_SHA256:
                raise ValueError("MobileNetV2 pretrained weight digest mismatch")
            temporary.replace(weights)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("MobileNetV2 pretrained weight digest mismatch")
    model = architecture()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def fixture(seed):
    """Deterministic normalized RGB fixture; accuracy on ImageNet is not measured."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    rgb = torch.rand((1, 3, 224, 224), generator=generator)
    mean = torch.tensor(MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(STD).reshape(1, 3, 1, 1)
    return ((rgb - mean) / std,)


def export(destination, *, offline=False, kernel_package=None, overwrite=False):
    from runtime.model_bundle import export_bundle
    print("Export MobileNetV2 1×3×224×224", flush=True)
    manifest = export_bundle(
        load(offline), fixture(20261025), Path(destination) / "mobilenetv2", model_id=MODEL_ID,
        model_revision=REVISION, input_names=["image"], output_names=["logits"], seed=20261025,
        additional_fixtures=[(seed, fixture(seed)) for seed in (20261026, 20261027, 20261125, 20261126, 20261127)],
        kernel_package=kernel_package, overwrite=overwrite,
        extra_files=[(ROOT / "models/TORCHVISION_LICENSE.txt", "licenses/TORCHVISION_LICENSE.txt")],
        preprocessing=dict(input_layout="NCHW", color="RGB", input_range=[0, 1], resize_shorter=256,
                           interpolation="bilinear", antialias=True, center_crop=[224, 224],
                           mean=list(MEAN), std=list(STD)),
        metadata=dict(weights_sha256=WEIGHTS_SHA256, weights_url=WEIGHTS_URL,
                      architecture_source="https://github.com/pytorch/vision/blob/v0.24.0/torchvision/models/mobilenetv2.py",
                      license="BSD-3-Clause", fixtures="seeded RGB inputs; reference correctness only"),
    )
    return {"mobilenetv2": manifest}
