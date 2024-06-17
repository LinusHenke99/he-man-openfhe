import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def main() -> None:
    x = np.load("./mnist_28x28_x.npy")

    x = x[0, 0]

    plt.imshow(x)
    plt.show()

    inp = input("Resize image? Y/n: ")

    if inp == "n":
        exit(0)

    resized = resize(x, (28, 28))

    resized = resized[None, :][None, :]

    filename = input("Enter filename: ")

    np.save(filename, resized)


if __name__ == "__main__":
    main()

