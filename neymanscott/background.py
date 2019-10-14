"""
Define various models for "background" observed events.
"""
import copy
import numpy as np
import numpy.random as npr
from numpy.linalg import inv, solve
from scipy.special import logsumexp, gammaln
from scipy.stats import invwishart, multivariate_normal, poisson, multinomial, expon
from scipy.optimize import linear_sum_assignment
from scipy.linalg import solve_triangular

from neymanscott.util import convex_combo

class Background(object):
    """
    Base class for background models
    """
    def initialize(self, data):
        pass

    def sample(self, size=1, **kwargs):
        raise NotImplementedError

    def log_likelihood(self, data):
        raise NotImplementedError

    def m_step(self, data, **kwargs):
        raise NotImplementedError

    def gibbs_step(self, data):
        raise NotImplementedError


class GaussianBackground(Background):
    """
    A wrapper for a Gaussian distribution
    """
    def __init__(self, data_dim=1, mu=None, Sigma=None,
                 mu0=None, lmbda0=None, Psi0=None, nu0=None):
        self.data_dim = data_dim

        # Set the NIW prior on background mean and covariance
        if mu0 is None:
            self.mu0 = np.zeros(self.data_dim)
        else:
            assert mu0.shape == (self.data_dim,)
            self.mu0 = mu0

        if lmbda0 is None:
            self.lmbda0 = 1
        else:
            assert np.isscalar(lmbda0)
            self.lmbda0 = lmbda0

        if Psi0 is None:
            self.Psi0 = 0.5 * np.eye(self.data_dim)
        else:
            assert Psi0.shape == (self.data_dim, self.data_dim)
            self.Psi0 = Psi0

        if nu0 is None:
            self.nu0 = self.data_dim + 2
        else:
            assert np.isscalar(nu0) and nu0 > self.data_dim - 1
            self.nu0 = nu0

        # Initialize the mean and covariance
        if mu is not None and Sigma is not None:
            assert mu.shape == (self.data_dim,)
            assert Sigma.shape == (self.data_dim, self.data_dim)
            self.mu = mu
            self.Sigma = Sigma

        else:
            self.Sigma = np.atleast_2d(invwishart(self.nu0, self.Psi0).rvs())
            assert np.all(np.linalg.eigvalsh(self.Sigma) > 0)
            self.mu = multivariate_normal(self.mu0, self.Sigma / self.lmbda0).rvs()

    def initialize(self, data):
        self.mu = np.mean(data, axis=0)
        self.Sigma = np.cov(data.T) + 1e-3 * np.eye(self.data_dim)

    def sample(self, count=1, size=1):
        if self.data_dim > 1:
            return np.atleast_2d(multivariate_normal.rvs(self.mu, self.Sigma, size=size))
        else:
            return np.atleast_1d(multivariate_normal.rvs(self.mu, self.Sigma, size=size))

    def log_likelihood(self, data):
        ll = multivariate_normal(self.mu, self.Sigma).logpdf(data)
        assert np.all(np.isfinite(ll))
        # assert ll.shape == (data.shape[0], )
        return ll

    def gibbs_step(self, background_data, **kwargs):
        """
        Perform a Gibbs update of the background parameters
        """
        N, D = background_data.shape
        if N > 0:
            nu_post = self.nu0 + N
            lmbda_post = self.lmbda0 + N
            bkgd_mean = background_data.mean(axis=0)
            bkgd_mean_diff = bkgd_mean - self.mu0
            bkgd_resid = background_data - bkgd_mean
            mu_post = (self.lmbda0 * self.mu0 + N * bkgd_mean) / lmbda_post
            bkgd_cov = bkgd_resid.T.dot(bkgd_resid)
            Psi_post = self.Psi0 + bkgd_cov + \
                self.lmbda0 * N / (self.lmbda0 + N) * np.outer(bkgd_mean_diff, bkgd_mean_diff)
        else:
            nu_post = self.nu0
            lmbda_post = self.lmbda0
            Psi_post = self.Psi0
            mu_post = self.mu0

        self.Sigma = invwishart(nu_post, Psi_post).rvs()
        self.mu = multivariate_normal(mu_post, self.Sigma / lmbda_post).rvs()

    def m_step(self, background_data, step_size=1.0, **kwargs):
        """
        Set the mean and variance to maximize expected log
        likelihood.

            theta* = argmax E_z[log p(y | z, theta)]

        where y is the observed data and z are the parents.
        The expectation wrt z is approximated with a sample.

        Take a convex combination of current parameters and
        optimal parameters for this sample.  Do the combination
        in information parameter space to get natural gradients.
        """
        N, D = background_data.shape
        assert D == self.data_dim
        if N == 0:
            # Sample from the prior
            self.Sigma = invwishart(self.nu0, self.Psi0).rvs()
            self.mu = multivariate_normal(self.mu0, self.Sigma / self.lmbda0).rvs()

        else:
            # Compute the posterior distribution over parameters given this minibatch
            nu_post = self.nu0 + N
            lmbda_post = self.lmbda0 + N
            bkgd_mean = background_data.mean(axis=0)
            bkgd_mean_diff = bkgd_mean - self.mu0
            bkgd_resid = background_data - bkgd_mean
            mu_post = (self.lmbda0 * self.mu0 + N * bkgd_mean) / lmbda_post
            bkgd_cov = bkgd_resid.T.dot(bkgd_resid)
            Psi_post = self.Psi0 + bkgd_cov + \
                self.lmbda0 * N / (self.lmbda0 + N) * np.outer(bkgd_mean_diff, bkgd_mean_diff)
            Sigma_post = Psi_post / (nu_post + self.data_dim + 1)

            # Take a convex combination of current and new parameters
            self.Sigma = convex_combo(self.Sigma, Sigma_post, step_size)
            self.mu = convex_combo(self.mu, mu_post, step_size)

            assert np.all(np.isfinite(self.Sigma))


class MultinomialBackground(Background):
    """
    A wrapper for a multinomial distribution
    """
    def __init__(self, data_dim=1, pi=None, concentration=1):
        self.data_dim = data_dim
        self.concentration = concentration

        assert pi is not None or concentration is not None, "Either pi or concentration must be specified."
        if pi is not None:
            assert pi.ndim == 1 and pi.size == data_dim and np.all(pi >= 0) and np.allclose(pi.sum(), 1)
        else:
            pi = npr.dirichlet(concentration * np.ones(data_dim))

        self.pi = pi

    def initialize(self, data):
        self.pi = np.sum(data, axis=0) + 1e-4
        self.pi /= self.pi.sum()

    def sample(self, count=1, size=1):
        return multinomial.rvs(count, self.pi, size=size)

    def log_likelihood(self, data):
        ll = multinomial.logpmf(data, data.sum(axis=-1), self.pi)
        assert np.isfinite(ll)
        return ll

    def m_step(self, data, **kwargs):
        # TODO: Decide if we want to actually update the background multinomial parameters
        pass


class DirichletMultinomialBackground(Background):
    """
    A wrapper for a dirichlet-multinomial distribution
    """
    def __init__(self, data_dim=1, concentration=1):
        self.concentration = concentration
        self.data_dim = data_dim

    def sample(self, count=1, size=1):
        pi = npr.dirichlet(self.concentration * np.ones(self.data_dim))
        return multinomial.rvs(count, pi, size=size)

    def log_likelihood(self, data):
        a = self.concentration * np.ones(self.data_dim)
        n = data.sum(axis=-1)
        pll = gammaln(n+1) + gammaln(a.sum()) - gammaln(n + a.sum())
        pll += np.sum(gammaln(data + a) - gammaln(data+1) - gammaln(a), axis=-1)
        assert np.isfinite(pll)
        return pll


class NodeIndexBackground(Background):
    """
    A wrapper for a discrete distribution
    """
    def __init__(self, pi, concentration=0.1):
        assert pi.ndim == 1 and np.all(pi >= 0) and np.allclose(pi.sum(), 1)
        self.num_nodes = len(pi)
        self.pi = pi
        self.log_pi = np.log(self.pi)
        self.concentration = concentration

    def initialize(self, data):
        pi = np.bincount(data, minlength=self.num_nodes) + 1e-4
        self.pi = pi / pi.sum()
        self.log_pi = np.log(self.pi)

    def sample(self, size=1):
        return npr.choice(self.num_nodes, p=self.pi, size=size)

    def log_likelihood(self, data):
        return self.log_pi[data]

    def m_step(self, data, step_size=1, **kwargs):
        pi_hat = np.bincount(data, minlength=self.num_nodes) + 1e-4
        pi_hat /= pi_hat.sum()

        self.pi = convex_combo(self.pi, pi_hat, step_size)
        assert np.allclose(self.pi.sum(), 1)
        self.log_pi = np.log(self.pi)

    def gibbs_step(self, data):
        a_post = np.bincount(data, minlength=self.num_nodes) + self.concentration
        self.pi = npr.dirichlet(a_post)
        self.log_pi = np.log(self.pi)


class UniformTimeBackground(Background):
    """
    Background distribution over uniformly distributed event times.
    """
    def __init__(self, T):
        assert T > 0
        self.T = T

    def sample(self, size=1):
        return npr.rand(size) * self.T

    def log_likelihood(self, data):
        return -np.log(self.T) * np.ones_like(data)

    def m_step(self, data, **kwargs):
        pass

    def gibbs_step(self, data):
        pass


class TimeAndMarkBackground(Background):
    def __init__(self, time_class, mark_class, time_kwargs={}, mark_kwargs={}):
        self.time_background = time_class(**time_kwargs)
        self.mark_background = mark_class(**mark_kwargs)

    def initialize(self, data):
        times, marks = data[:, 0], data[:, 1:]
        self.time_background.initialize(times)
        self.mark_background.initialize(marks)

    def sample(self, size=1, time_kwargs={}, mark_kwargs={}):
        times = self.time_background.sample(size=size, **time_kwargs)
        marks = self.mark_background.sample(size=size, **mark_kwargs)
        return np.column_stack((times, marks))

    def log_likelihood(self, data):
        if data.ndim == 1:
            time = data[0]
            mark = data[1:]
            ll = self.time_background.log_likelihood(time)
            ll += self.mark_background.log_likelihood(mark)
        else:
            time = data[:, 0]
            mark = data[:, 1:]

            ll = self.time_background.log_likelihood(time)
            ll += self.mark_background.log_likelihood(mark)
            assert ll.shape == (data.shape[0],)

        return ll

    def m_step(self, data, **kwargs):
        times, marks = data[:, 0], data[:, 1:]
        self.time_background.m_step(times, **kwargs)
        self.mark_background.m_step(marks, **kwargs)

    def gibbs_step(self, data):
        times, marks = data[:, 0], data[:, 1:]
        self.time_background.gibbs_step(times)
        self.mark_background.gibbs_step(marks)


class NodeAndTimeBackground(Background):
    def __init__(self, num_nodes, node_distribution,
                 time_class, time_kwargs={}):
        self.num_nodes = num_nodes
        self.node_background = NodeIndexBackground(node_distribution)
        self.time_background = time_class(**time_kwargs)

    def initialize(self, data):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        self.node_background.initialize(nodes)
        self.time_background.initialize(times)

    def sample(self, size=1, time_kwargs={}, mark_kwargs={}):
        nodes = self.node_background.sample(size=size)
        times = self.time_background.sample(size=size, **time_kwargs)
        return np.column_stack((nodes, times))

    def log_likelihood(self, data):
        node = data[0].astype(int)
        time = data[1]
        ll = self.node_background.log_likelihood(node)
        assert np.all(np.isfinite(ll))
        ll += self.time_background.log_likelihood(time)
        assert np.all(np.isfinite(ll))
        return ll

    def m_step(self, data, **kwargs):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        self.node_background.m_step(nodes, **kwargs)
        self.time_background.m_step(times, **kwargs)

    def gibbs_step(self, data):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        self.node_background.gibbs_step(nodes)
        self.time_background.gibbs_step(times)


class FactorizedNodeAndTimeAndMarkBackground(Background):
    def __init__(self, num_nodes, node_distribution,
                 time_class, mark_class,
                 time_kwargs={}, mark_kwargs={}):
        self.num_nodes = num_nodes
        self.node_background = NodeIndexBackground(node_distribution)
        self.time_background = time_class(**time_kwargs)
        self.mark_background = mark_class(**mark_kwargs)

    def initialize(self, data):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        marks = data[:, 2:]
        self.node_background.initialize(nodes)
        self.time_background.initialize(times)
        self.mark_background.initialize(marks)

    def sample(self, size=1, time_kwargs={}, mark_kwargs={}):
        nodes = self.node_background.sample(size=size)
        times = self.time_background.sample(size=size, **time_kwargs)
        marks = self.mark_background.sample(size=size, **mark_kwargs)
        return np.column_stack((nodes, times, marks))

    def log_likelihood(self, data):
        node = data[0].astype(int)
        time = data[1]
        mark = data[2:]
        ll = self.node_background.log_likelihood(node)
        ll += self.time_background.log_likelihood(time)
        ll += self.mark_background.log_likelihood(mark)
        return ll

    def m_step(self, data, **kwargs):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        marks = data[:, 2:]
        self.node_background.m_step(nodes, **kwargs)
        self.time_background.m_step(times, **kwargs)
        self.mark_background.m_step(marks, **kwargs)

    def gibbs_step(self, data):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        marks = data[:, 2:]
        self.node_background.gibbs_step(nodes)
        self.time_background.gibbs_step(times)
        self.mark_background.gibbs_step(marks)


class NodeAndTimeAndMarkBackground(Background):
    def __init__(self, num_nodes, node_distribution,
                 time_class, mark_class,
                 time_kwargs={}, mark_kwargs={}):
        self.num_nodes = num_nodes
        self.node_background = NodeIndexBackground(node_distribution)
        self.time_background = time_class(**time_kwargs)
        self.mark_backgrounds = [mark_class(**mark_kwargs) for _ in range(num_nodes)]

    def initialize(self, data):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        marks = data[:, 2:]
        self.node_background.initialize(nodes)
        self.time_background.initialize(times)
        for n in range(self.num_nodes):
            self.mark_backgrounds[n].initialize(marks[nodes == n])

    def sample(self, size=1, time_kwargs={}, mark_kwargs={}):
        nodes = self.node_background.sample(size=size)
        times = self.time_background.sample(size=size, **time_kwargs)
        marks = np.concatenate([
            self.mark_backgrounds[m].sample(size=1, **mark_kwargs)
            for m in nodes])
        return np.column_stack((nodes, times, marks))

    def log_likelihood(self, data):
        if data.ndim == 1:
            node = data[0].astype(int)
            time = data[1]
            mark = data[2:]
            ll = self.node_background.log_likelihood(node)
            ll += self.time_background.log_likelihood(time)
            ll += self.mark_backgrounds[node].log_likelihood(mark)
        else:
            node = data[:, 0].astype(int)
            time = data[:, 1]
            mark = data[:, 2:]

            ll = self.node_background.log_likelihood(node)
            ll += self.time_background.log_likelihood(time)
            ll += np.array([self.mark_backgrounds[n].log_likelihood(m) for n, m in zip(node, mark)])
            assert ll.shape == (data.shape[0],)

        return ll

    def m_step(self, data, **kwargs):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        marks = data[:, 2:]
        self.node_background.m_step(nodes, **kwargs)
        self.time_background.m_step(times, **kwargs)
        for n in range(self.num_nodes):
            self.mark_backgrounds[n].m_step(marks[nodes == n], **kwargs)

    def gibbs_step(self, data):
        nodes = data[:, 0].astype(int)
        times = data[:, 1]
        marks = data[:, 2:]
        self.node_background.gibbs_step(nodes)
        self.time_background.gibbs_step(times)
        for n in range(self.num_nodes):
            self.mark_backgrounds[n].gibbs_step(marks[nodes == n])


class SeqNMFBackground(Background):
    """
    Background for SeqNMF model
    """
    def __init__(self, T, num_neurons, pi_bkgd):
        self.T = T
        self.N = num_neurons
        assert pi_bkgd.shape == (self.N,) and np.all(pi_bkgd >= 0) and np.allclose(pi_bkgd.sum(), 1.0)
        self.pi_bkgd = pi_bkgd

    def initialize(self, data):
        counts = np.bincount(data[:,0].astype(int), minlength=self.N)
        self.pi_bkgd = npr.dirichlet(np.ones(self.N) + counts)

    def sample(self, size=1, **kwargs):
        ns = npr.choice(self.N, p=self.pi_bkgd, size=size)
        ts = npr.rand(size) * self.T
        return np.column_stack((ns, ts))

    def log_likelihood(self, data):
        return -np.log(self.T) + np.log(self.pi_bkgd)[data[0].astype(int)]

    def m_step(self, data, **kwargs):
        counts = np.bincount(data[:,0].astype(int), minlength=self.N)
        self.pi_bkgd = counts + 1e-1
        self.pi_bkgd /= self.pi_bkgd.sum()

    def gibbs_step(self, data):
        counts = np.bincount(data[:,0].astype(int), minlength=self.N)
        self.pi_bkgd = npr.dirichlet(1e-1 * np.ones(self.N) + counts)

