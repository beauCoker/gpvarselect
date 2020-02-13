import numpy as np


__all__ = [
    'x3_toy',
    'x3_gap_toy',
    'sin_toy',
]


class toy_dataset(object):
    def __init__(self, name=''):
        self.name = name

    def train_samples(self):
        raise NotImplementedError

    def test_samples(self):
        raise NotImplementedError


class x3_toy(toy_dataset):
    def __init__(self, name='x3'):
        self.x_min = -4
        self.x_max = 4
        #self.y_min = -100
        #self.y_max = 100
        #self.confidence_coeff = 3.
        self.f = lambda x: np.power(x, 3)
        self.y_std = 3.
        super(x3_toy, self).__init__(name)

    def train_samples(self, n_data=20, seed=2):
        np.random.seed(seed)
        X_train = np.random.uniform(self.x_min, self.x_max, (n_data, 1))
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(X_train ** 3 + epsilon)
        return X_train, y_train

    def test_samples(self, n_data = 1000, seed=2):
        np.random.seed(seed)
        X_test = np.linspace(self.x_min, self.x_max, num=n_data)
        epsilon = np.random.normal(0, self.y_std, n_data)
        y_test = np.squeeze(X_test ** 3 + epsilon)
        return X_test, y_test

class sin_toy(toy_dataset):
    def __init__(self, name='sin'):
        self.x_min = -1
        self.x_max = 1
        #self.y_min = -3.5
        #self.y_max = 3.5
        #self.confidence_coeff = 1.
        #self.y_std = 2e-1
        self.y_std = np.sqrt(.04)

        def f(x):
            return 2 * np.sin(4*x)
        self.f = f
        super(sin_toy, self).__init__(name)

    def train_samples(self, n_data=20, seed=3):
        np.random.seed(seed)

        X_train = np.random.uniform(self.x_min, self.x_max, (n_data, 1))
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(self.f(X_train) + epsilon)
        return X_train, y_train

    def test_samples(self, n_data=20, seed=3):
        np.random.seed(seed)

        X_test = np.linspace(self.x_min, self.x_max, n_data).reshape(n_data, 1)
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_test = np.squeeze(self.f(X_test) + epsilon)
        return X_test, y_test

class rbf_toy2(toy_dataset):
    def __init__(self, variance=1.0, lengthscale=1.0, name='rbf'):
        self.variance = variance
        self.lengthscale = lengthscale
        self.x_min = 0
        self.x_max = 1
        #self.y_min = -50
        #self.y_max = 50
        #self.confidence_coeff = np.sqrt(.1)
        self.y_std = np.sqrt(.001)
        super(rbf_toy2, self).__init__(name)

    def sample_f(self, n_train_max, n_test, seed=0):
        self.X_train_max = np.random.uniform(self.x_min, self.x_max, (n_train_max, 1))
        self.X_test = np.linspace(self.x_min, self.x_max, n_test).reshape(-1,1)
        X = np.concatenate([self.X_train_max, self.X_test], axis=0)
        K = self.variance*np.exp(-0.5*(X.reshape(-1,1)-X.reshape(1,-1))**2 / self.lengthscale**2)
        f = np.random.multivariate_normal(np.zeros(X.shape[0]), K)
        self.f_train_max = f[:n_train_max].reshape(-1,1)
        self.f_test = f[-n_test:].reshape(-1,1)

    def train_samples(self, n_data, seed=0):
        np.random.seed(seed)
        idx_sample = np.random.choice(self.X_train_max.shape[0], n_data)
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = self.f_train_max[idx_sample] + epsilon
        return self.X_train_max[idx_sample], y_train 

    def test_samples(self, n_data=None, seed=0):
        np.random.seed(seed)
        epsilon = np.random.normal(0, self.y_std, (self.f_test.shape[0], 1))
        y_test = self.f_test + epsilon
        return self.X_test, y_test


class rbf_toy(toy_dataset):
    def __init__(self, variance=1.0, lengthscale=1.0, name='rbf'):
        self.variance = variance
        self.lengthscale = lengthscale
        self.x_min = 0
        self.x_max = 1
        #self.y_min = -50
        #self.y_max = 50
        #self.confidence_coeff = np.sqrt(.1)
        self.y_std = np.sqrt(.1)
        super(rbf_toy, self).__init__(name)

    def sample_f(self, n_train, n_test, seed=0):
        self.X_train = np.random.uniform(-4, 4, (n_train, 1))
        self.X_test = np.linspace(self.x_min, self.x_max, n_test).reshape(-1,1)
        X = np.concatenate([self.X_train, self.X_test], axis=0)
        K = self.variance*np.exp(-0.5*(X.reshape(-1,1)-X.reshape(1,-1))**2 / self.lengthscale**2)
        f = np.random.multivariate_normal(np.zeros(X.shape[0]), K)
        self.f_train = f[:n_train].reshape(-1,1)
        self.f_test = f[-n_test:].reshape(-1,1)

    def train_samples(self, seed=0):
        np.random.seed(seed)
        epsilon = np.random.normal(0, self.y_std, (self.f_train.shape[0], 1))
        y_train = self.f_train + epsilon
        return self.X_train, y_train 

    def test_samples(self, seed=0):
        np.random.seed(seed)
        epsilon = np.random.normal(0, self.y_std, (self.f_test.shape[0], 1))
        y_test = self.f_test + epsilon
        return self.X_test, y_test


class comp_toy(toy_dataset):
    def __init__(self, name='comp'):
        self.x_min = -1
        self.x_max = 1
        #self.y_min = -6
        #self.y_max = 6
        #self.confidence_coeff = 1.
        self.y_std = np.sqrt(.04)

        def f(x):
            return (1+x) * np.sin(10*np.tanh(x))
        self.f = f
        super(comp_toy, self).__init__(name)

    def train_samples(self, n_data=20, seed=3):
        np.random.seed(seed)

        X_train = np.random.uniform(self.x_min, self.x_max, (n_data, 1))
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(self.f(X_train) + epsilon)
        return X_train, y_train

    def test_samples(self, n_data=20, seed=3):
        np.random.seed(seed)

        X_train = np.linspace(self.x_min, self.x_max, n_data).reshape(n_data, 1)
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(self.f(X_train) + epsilon)
        return X_train, y_train


class x3_gap_toy(toy_dataset):
    def __init__(self, name='x3_gap'):
        self.x_min = -6
        self.x_max = 6
        self.y_min = -100
        self.y_max = 100
        self.confidence_coeff = 3.
        self.y_std = 3.
        self.f = lambda x: np.power(x, 3)
        super(x3_gap_toy, self).__init__(name)

    def train_samples(self, n_data=20):
        np.random.seed(1)

        X_train_1 = np.random.uniform(-4, -1, (n_data // 2, 1))
        X_train_2 = np.random.uniform(1, 4, (n_data // 2, 1))
        X_train = np.concatenate([X_train_1, X_train_2], axis=0)
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(X_train ** 3 + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(-6, 6, num=1000, dtype=np.float32)
        outputs = np.power(inputs, 3)
        return inputs, outputs

'''
class sin_toy(toy_dataset):
    def __init__(self, name='sin'):
        self.x_min = -5
        self.x_max = 5
        self.y_min = -3.5
        self.y_max = 3.5
        self.confidence_coeff = 1.
        self.y_std = 2e-1

        def f(x):
            return 2 * np.sin(4*x)
        self.f = f
        super(sin_toy, self).__init__(name)

    def train_samples(self, n_data=20):
        np.random.seed(3)

        X_train1 = np.random.uniform(-2, -0.5, (int(n_data/2), 1))
        X_train2 = np.random.uniform(0.5, 2, (int(n_data/2), 1))
        X_train = np.concatenate([X_train1, X_train2], axis=0)
        epsilon = np.random.normal(0, self.y_std, (2*int(n_data/2), 1))
        y_train = np.squeeze(self.f(X_train) + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(self.x_min, self.x_max, 1000)
        return inputs, self.f(inputs)
'''