from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPOSITORY = "MichaelMukiibi/fastdrs-models"
DEFAULT_VERSION = "v0.1.0"


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    architecture: str
    num_classes: int
    img_size: int
    filename: str
    sha256: str | None


def get_cache_dir() -> Path:
    """
    Return the fastDRS model cache directory.
    """

    return (
        Path.home()
        / ".cache"
        / "fastdrs"
        / "models"
    )


def load_registry(
    registry_url: str | None = None,
) -> dict:
    """
    Load the model registry.

    By default, the registry is loaded from the fastDRS model
    repository on GitHub.
    """

    if registry_url is None:
        registry_url = (
            "https://raw.githubusercontent.com/"
            f"{DEFAULT_REPOSITORY}/main/models.json"
        )

    with urllib.request.urlopen(registry_url) as response:
        return json.loads(response.read())


def get_model_artifact(
    model_name: str,
    backend: str = "pytorch",
    version: str = DEFAULT_VERSION,
) -> ModelArtifact:
    """
    Get metadata for a registered model artifact.
    """

    if backend not in {"pytorch", "litert"}:
        raise ValueError(
            "backend must be 'pytorch' or 'litert'."
        )

    registry = load_registry()

    models = registry.get("models", {})

    if model_name not in models:
        available = ", ".join(sorted(models))

        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    model = models[model_name]
    artifact = model.get(backend)

    if artifact is None:
        raise ValueError(
            f"Model '{model_name}' does not have "
            f"a {backend} artifact."
        )

    return ModelArtifact(
        name=model_name,
        architecture=model["architecture"],
        num_classes=model["num_classes"],
        img_size=model["img_size"],
        filename=artifact["filename"],
        sha256=artifact.get("sha256"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_checksum(
    path: Path,
    expected: str | None,
) -> None:
    if expected is None:
        return

    actual = _sha256(path)

    if actual != expected:
        path.unlink(missing_ok=True)

        raise RuntimeError(
            f"Checksum verification failed for "
            f"{path.name}.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def download_model(
    model_name: str,
    backend: str = "pytorch",
    version: str = DEFAULT_VERSION,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download and cache a fastDRS model artifact.

    Parameters
    ----------
    model_name:
        Registered model name.

    backend:
        Either "pytorch" or "litert".

    version:
        GitHub Release version.

    cache_dir:
        Optional custom cache directory.

    force:
        Force a fresh download.

    Returns
    -------
    pathlib.Path
        Local path to the downloaded model.
    """

    artifact = get_model_artifact(
        model_name=model_name,
        backend=backend,
        version=version,
    )

    if cache_dir is None:
        cache = get_cache_dir()
    else:
        cache = Path(cache_dir)

    cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = cache / artifact.filename

    if destination.exists() and not force:
        _verify_checksum(
            destination,
            artifact.sha256,
        )

        return destination

    url = (
        f"https://github.com/"
        f"{DEFAULT_REPOSITORY}/releases/download/"
        f"{version}/"
        f"{artifact.filename}"
    )

    temporary = destination.with_suffix(
        destination.suffix + ".download"
    )

    try:
        urllib.request.urlretrieve(
            url,
            temporary,
        )

        _verify_checksum(
            temporary,
            artifact.sha256,
        )

        temporary.replace(destination)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def list_pretrained_models() -> list[str]:
    """
    Return all registered pretrained model names.
    """

    registry = load_registry()

    return sorted(
        registry.get("models", {}).keys()
    )