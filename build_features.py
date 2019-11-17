from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit


def load_moons(samples=3000, features=2, noise=0.25):
    dataset_name="Moons"

    x, y = make_moons(n_samples=samples, noise=noise)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train, x_test, y_train, y_test


def load_concentric(seed=42, samples=3000, feature=2, noise=0.25, factor=0.5):
    x, y = make_circles(n_samples=samples, noise=0.25, factor=factor)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=seed)

    return x_train, x_test, y_train, y_test