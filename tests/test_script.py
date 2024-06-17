from test_inference import test_mnist_relu_inference
from test_inference import test_inference
from pathlib import Path
from shutil import rmtree
from os import mkdir


def clean(path: Path) -> None:
    if path.exists():
        rmtree(path)
        mkdir(path)


def main() -> None:
    path = Path("./dump")

    clean(path)

    test_inference(path)

    clean(path)

    test_mnist_relu_inference(path)


if __name__ == "__main__":
    main()
