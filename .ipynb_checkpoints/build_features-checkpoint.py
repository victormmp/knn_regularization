from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit


def load_moons(seed=42, samples=3000, features=2):
    dataset_name="Moons"

    x, y = make_moons(n_samples=samples, random_state=seed, noise=0.25)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=seed)

    return x_train, x_test, y_train, y_test