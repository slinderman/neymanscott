"""
Define different type of conjugate cluster observation models.
"""
import copy
import numpy as np
import numpy.random as npr
from scipy.special import logsumexp, gammaln
from scipy.stats import invwishart, multivariate_normal, poisson, multinomial, expon, norm, dirichlet
from scipy.optimize import linear_sum_assignment
from scipy.linalg import solve_triangular


class Cluster(object):
    """
    Holds the sufficient statistics for a cluster.
    Must support adding/removing data and evaluating
    the predictive log likelihood of a new datapoint.
    """
    def __init__(self, **hypers):
        self.size = 0
        self._params = None

    def add_datapoint(self, x):
        self.size += 1

    def remove_datapoint(self, x):
        self.size -= 1
        assert self.size >= 0

    @property
    def params(self):
        return self._params

    @property
    def posterior_mode(self):
        pass

    def sample(self, size=1):
        pass

    def jitter(self):
        """
        Return a copy of this cluster with jittered parameters.
        Also return the forward and reverse log probabilities of
        the jitter move.
        """
        raise NotImplementedError

    def log_likelihood(self, data):
        """
        Compute the log likelihood of data under current cluster parameters
        """
        raise NotImplementedError

    def log_normalizer(self):
        """
        Compute the log normalizer.  In an exponential family,

          p(u | eta) = h(u) * exp(t(u) * eta - log Z(eta))

        where eta are the natural parameters and log Z(eta) is
        the log normalizer.  It is useful for computing the marginal
        likelihood of a dataset and the predictive likelihood of
        a new datapoint.
        """
        pass

    def predictive_log_likelihood(self, x):
        """
        Assume the likelihood is conjugate with the prior on u in that

            p(x | u) = exp(s(x) * t(u) - log F(x))

        where s(x) are the sufficient statistics of x and t(u) is
        as above. Then the predictive only depends on the difference
        in log normalizers of the prior.

        Let x denote the new datapoint and X the set of datapoints
        already in the cluster. We have,

            p(x | X, eta) = p(x, X | eta) / p(X | eta)

        Both the numerator and denominator are marginal densities,
        integrating out the cluster parameters.  That is,

            p(X | eta) = int p(X, u | eta) du
                = int prod_n[p(xn | u)] p(u | eta) du
                = prod_n[1 / F(xn)] Z(eta + sum_n s(xn)) / Z(eta)

        The predictive likelihood is thus,

            p(x | X, eta) = Z(eta + sum_n s(xn) + s(x)) / Z(eta + sum_n s(xn)) / F(x)

        If we are comparing predictive likelihoods across clusters with the
        same likelihood model, F(x) will be a constant.  We assume this is
        the case and drop it.
        """
        logZ_curr = self.log_normalizer()
        self.add_datapoint(x)
        logZ_new = self.log_normalizer()
        self.remove_datapoint(x)

        # Cache the old value again
        self._log_normalizer = logZ_curr
        self._stale = False

        return logZ_new - logZ_curr


class ClusterFactory(object):
    """
    Class that stores hyperparameters of the clusters.
    """
    def make_new_cluster(self):
        raise NotImplementedError

    def gibbs_step(self, data, parents):
        raise NotImplementedError

    def m_step(self, data, parents):
        raise NotImplementedError


class DefaultClusterFactory(ClusterFactory):
    """
    By default, treat the cluster hyperparameters as fixed.
    """
    def __init__(self, cluster_class, cluster_hypers):
        self.cluster_class = cluster_class
        self.cluster_hypers = cluster_hypers

    def make_new_cluster(self):
        return self.cluster_class(**self.cluster_hypers)

    def gibbs_step(self, clusters, data, parents):
        pass

    def m_step(self, data, parents):
        pass


class GaussianCluster(Cluster):
    """
    Cluster with Gaussian observations and a normal-inverse Wishart (NIW) prior.

    The NIW prior has standard parameters (mu0, lambda, Psi, nu).
    See https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution

    Its density is given by the product of a normal and an inverse Wishart,

    p(mu, Sigma) = N(mu | mu0, 1/lambda Sigma) * IW(Sigma | Psi, nu)
       = (2 pi)^{-D/2} |Sigma|^{-1/2} lambda^{D / 2}
        * exp(-lambda/2 * (mu - mu0)^T Sigma^{-1} (mu - mu0))
        * |Psi|^{nu/2} * 2^{-nu * D/2} / Gam_D(nu/2)
        * |Sigma|^{-(nu + D + 1) / 2}
        * exp(-1/2 Tr(Psi Sigma^{-1}))

       =  exp(-(nu+D+2)/2 * log |Sigma|)
        * exp(-1/2 lambda * mu^T Sigma^{-1} mu)
        * exp(     lambda * mu^T Sigma^{-1} mu0)
        * exp(-1/2 Tr((Psi + lambda * mu0 mu0^T) Sigma^{-1})
        / Z(eta)

    where Z(eta) = 2pi^{D/2} * lambda^{-D / 2} * |Psi|^{-nu/2} * 2^{nu * D/2} * Gam_D(nu/2)

    From this we can read off the sufficient statistics:

    eta = (-1/2 log |Sigma|,                # call this n1
           -1/2 mu^T Sigma^{-1} mu,         # call this n2
           mu^T Sigma^{-1},                 # call this h
           -1/2 Sigma^{-1})                 # call this J

    The prior's natural parameters are,
        (nu + D + 2, lambda, lambda * mu0, Psi + lambda * mu0 mu0^T)

    A single observation x under a Gaussian likelihood, N(x | mu, Sigma),
    looks like,

    log p(x | mu, Sigma)
        = -D/2 log 2pi -1/2 log |Sigma| -1/2 (x - mu)^T Sigma^{-1} (x - mu)

    which contributes sufficient statistics, (1, 1, x, xx^T).
    """
    def __init__(self, mu0=None, lmbda=None, Psi=None, nu=None):
        """
        Store the natural parameters of the NIW distribution.
        """
        super(GaussianCluster, self).__init__()

        # Validate inputs
        assert mu0.ndim == 1
        self.data_dim = mu0.shape[0]
        assert np.isscalar(lmbda)
        assert Psi.shape == (self.data_dim, self.data_dim)
        assert np.isscalar(nu) and nu > self.data_dim - 1

        # Store the natural parameters
        self.n1, self.n2, self.h, self.J = \
            self._standard_to_natural(mu0, lmbda, Psi, nu)

        # Sample the cluster parameters
        mu0, lmbda, Psi, nu = \
            self._natural_to_standard(self.n1, self.n2, self.h, self.J)
        Sigma = invwishart(nu, Psi).rvs()
        mu = multivariate_normal(mu0, Sigma / lmbda).rvs()
        self._params = (mu, Sigma)

        # Keep track of whether or not the stats are up to date
        self._stale = True
        self._log_normalizer = np.inf

    def _standard_to_natural(self, mu0, lmbda, Psi, nu):
        return (nu + self.data_dim + 2,
                lmbda,
                lmbda * mu0,
                Psi + lmbda * np.outer(mu0, mu0))

    def _natural_to_standard(self, n1, n2, h, J):
        return h / n2, n2, J - np.outer(h, h) / n2, n1 - self.data_dim - 2

    def log_prior(self):
        mu0, lmbda, Psi, nu = \
            self._natural_to_standard(self.n1, self.n2, self.h, self.J)
        mu, Sigma = self.params

        return invwishart(nu, Psi).logpdf(Sigma) + multivariate_normal(mu0, Sigma / lmbda).logpdf(mu)

    def log_likelihood(self, data):
        return multivariate_normal(*self.params).logpdf(data)

    @property
    def posterior_mode(self):
        mu0, lmbda, Psi, nu = \
            self._natural_to_standard(self.n1, self.n2, self.h, self.J)
        return mu0, Psi / (nu + self.data_dim + 1)

    def sample_posterior(self):
        mu0, lmbda, Psi, nu = \
            self._natural_to_standard(self.n1, self.n2, self.h, self.J)
        Sigma = invwishart(nu, Psi).rvs()
        mu = multivariate_normal(mu0, Sigma / lmbda).rvs()
        self._params = (mu, Sigma)

    def add_datapoint(self, x):
        super(GaussianCluster, self).add_datapoint(x)
        self.n1 += 1
        self.n2 += 1
        self.h += x
        self.J += np.outer(x, x)
        self.sample_posterior()
        self._stale = True

    def remove_datapoint(self, x):
        super(GaussianCluster, self).remove_datapoint(x)
        self.n1 -= 1
        self.n2 -= 1
        self.h -= x
        self.J -= np.outer(x, x)
        self.sample_posterior()
        self._stale = True

    def sample(self, size=1):
        if self.data_dim > 1:
            return np.atleast_2d(multivariate_normal(*self.params).rvs(size))
        else:
            return np.atleast_1d(multivariate_normal(*self.params).rvs(size))

    def jitter(self, mu_std=0.1, Sigma_dof=3):
        mu, Sigma = self.params
        new_mu = mu + mu_std * npr.randn(self.data_dim)
        new_Sigma = invwishart(self.data_dim + Sigma_dof, (Sigma_dof - 1) * Sigma).rvs()
        fwd_lp = rev_lp = multivariate_normal.logpdf(new_mu, mu, mu_std**2 * np.eye(self.data_dim))
        fwd_lp += invwishart(self.data_dim + Sigma_dof, (Sigma_dof - 1) * Sigma).logpdf(new_Sigma)
        rev_lp += invwishart(self.data_dim + Sigma_dof, (Sigma_dof - 1) * new_Sigma).logpdf(Sigma)

        new_cluster = copy.deepcopy(self)
        new_cluster._params = (new_mu, new_Sigma)
        return new_cluster, fwd_lp, rev_lp

    def log_normalizer(self):
        """
        Compute the log normalizer of the NIW distribution.

        Z(eta) = 2pi^{D/2} * lambda^{-D / 2}
                 * |Psi|^{-nu/2} * 2^{nu * D/2} * Gam_D(nu/2)

        where Gam_D(nu/2) = pi^{D*(D-1)/4} * prod_{d=1}^D Gam(nu/2 + (1-d)/2)
        """
        if self._stale:
            # Recompute the log normalizer
            D = self.data_dim
            mu0, lmbda, Psi, nu = \
                self._natural_to_standard(self.n1, self.n2, self.h, self.J)

            logZ = -0.5 * D * np.log(lmbda)
            logZ -= 0.5 * nu * np.linalg.slogdet(Psi)[1]
            logZ += np.sum(gammaln(0.5 * (nu - np.arange(D))))
            logZ += 0.5 * D * nu * np.log(2)
            # Extra constants
            logZ += 0.5 * D * np.log(2 * np.pi)
            logZ += (D * (D-1) / 4) * np.log(np.pi)
            self._log_normalizer = logZ
            self._stale = False

        return self._log_normalizer

    def predictive_log_likelihood(self, x):
        """
        Directly compute the predictive log likelihood.

        It's a multivariate Student's t distribution.
        """
        D = self.data_dim
        mu0, lmbda, Psi, nu = \
            self._natural_to_standard(self.n1, self.n2, self.h, self.J)

        Sigma = (lmbda + 1) / (lmbda * (nu - D + 1)) * Psi
        xc = np.array(x - mu0, ndmin=2)
        L = np.linalg.cholesky(Sigma)
        xs = solve_triangular(L, xc.T, overwrite_b=True, lower=True)
        lps = gammaln((nu + D) / 2.) - gammaln(nu / 2.) \
              - (D / 2.) * np.log(nu * np.pi) - np.log(L.diagonal()).sum() \
              - (nu + D) / 2. * np.log1p(1. / nu * np.sum(xs**2, axis=0))

        return lps[0] if lps.size == 1 else lps


class MultinomialCluster(Cluster):
    """
    Cluster with multinomial observations and a Dirichlet prior.

    The Dirichlet(a) prior density is

        p(u | a) = Gam(sum_k ak) / prod_k Gam(ak) * prod_k uk^{ak-1}

    The sufficient statistics are log(uk) for k = 1, .., K.

    The multinomial likelihood is,

        p(x | u) = Gam(sum_k xk + 1) / prod_k Gam(x_k + 1) prod_k uk^xk

    So that the posterior is

        p(u | x, a) = Dir(a1 + x1, ..., aK + xK)

    """
    def __init__(self, data_dim=None, concentration=None):
        super(MultinomialCluster, self).__init__()

        # Validate hyperparameters
        assert isinstance(data_dim, int) and data_dim > 1
        assert (np.isscalar(concentration) and concentration > 0) \
            or (concentration.shape == (data_dim,) and np.all(concentration > 0))
        self.data_dim = data_dim
        self.concentration = concentration

        # Store the natural parameters of the posterior
        self.a = concentration * np.ones(data_dim)
        pi = npr.dirichlet(self.a)
        self._params = (pi,)

        # Keep track of whether or not the stats are up to date
        self._stale = True
        self._log_normalizer = np.inf

    @property
    def posterior_mode(self):
        return self.a / self.a.sum()

    def sample_posterior(self):
        pi = npr.dirichlet(self.a)
        self._params = (pi,)

    def log_prior(self):
        return dirichlet(self.a).logpdf(self.params[0])

    def log_likelihood(self, data):
        log_pi = np.log(self.params[0])
        return np.sum(data * log_pi, axis=-1)

    def add_datapoint(self, x):
        super(MultinomialCluster, self).add_datapoint(x)
        assert x.shape == (self.data_dim,)
        self.a += x
        self.sample_posterior()
        self._stale = True

    def remove_datapoint(self, x):
        super(MultinomialCluster, self).remove_datapoint(x)
        assert x.shape == (self.data_dim,)
        self.a -= x
        self.sample_posterior()
        self._stale = True

    def sample(self, count=1, size=1):
        # Sample Dirichlet then multinomial
        return np.atleast_2d(npr.multinomial(count, *self.params, size=size))

    def jitter(self, concentration=100):
        pi = self.params[0]
        new_pi = npr.dirichlet(concentration * pi) + 1e-8
        new_pi /= new_pi.sum()
        fwd_lp = dirichlet(concentration * pi).logpdf(new_pi)
        rev_lp = dirichlet(concentration * new_pi).logpdf(pi)

        new_cluster = copy.deepcopy(self)
        new_cluster._params = (new_pi,)
        return new_cluster, fwd_lp, rev_lp

    def log_normalizer(self):
        """
        Compute the log normalizer of the NIW distribution.

        Z(eta) = prod_k Gam(ak) / Gam(sum_k ak)
        """
        if self._stale:
            # Recompute the log normalizer
            self._log_normalizer = np.sum(gammaln(self.a)) - gammaln(np.sum(self.a))
            self._stale = False

        return self._log_normalizer

    def predictive_log_likelihood(self, x):
        """
        Explicitly calculate the predictive log likelihood with
        all its normalizing constants.  Note that this only involves
        the nonzero entries in x.
        """
        a = self.a
        n = x.sum(axis=-1)
        pll = gammaln(n+1) + gammaln(a.sum()) - gammaln(n + a.sum())
        pll += np.sum(gammaln(x + a) - gammaln(x+1) - gammaln(a), axis=-1)
        # if issparse(x):
        #     indices, values = x.indices, x.values
        #     pll += np.sum(gammaln(values + self.a[indices]) - gammaln(values+1) - gammaln(self.a[indices]))
        return pll



class ExponentialTimeCluster(Cluster):
    """
    Cluster with time-stamped observations.  The cluster is parameterzied
    by a start time with a uniform prior on [0, T].  Observed event times
    are exponentially distributed after t_start.

    The posterior distribution of the start time s given child event times
    {t_n} is

        p(s | {t_n}) \propto Unif(s | [0, T]) * \prod_n Exp(t_n - s | tau)
            = 1/T * I[s \in [0, T]] * \prod_n 1/tau exp{-(t_n -s) / tau} * I[t_n > s]
            = 1/T * I[s \in [0, T]] * 1/tau^N exp{-\sum_n t_n / tau + s * N / tau} * I[s < min {t_n}]
            \propto I[s \in [0, min {t_n}]] * 1/(tau / N) exp{s / (tau / N)}

    Let u = min {t_n} - s.  Then,

        p(u | {t_n}) = Exp(u | tau / N) * I[ds \in [0, min {t_n}]]

    This is a truncated exponential random variable.  We can typically
    ignore the truncation when min {t_n} >> 0.  That makes the next integral
    easier.

    The posterior predictive, after a bit of math, is

        p(t' | {t_n}) = N/(N+1) Exp(t' - t_min | tau)   if t' > t_min
                     or 1/(N+1) Exp(t_min - t' | tau/N) if t' < t_min

    where t_min = min_n {t_n} and N is the number of points already in the cluster.
    """
    def __init__(self, T=1, tau=1):

        super(ExponentialTimeCluster, self).__init__()

        # Validate hyperparameters
        assert T > 0
        self.T = T
        assert tau > 0
        self.tau = tau

        # Sufficient statistic is min {t_n}, but in order to
        # update the cluster we need the whole array of times
        # self.ts = np.array([])

        # Hopefully faster: don't keep a list of events, just
        # store the min time and raise an exception if we remove
        # the min event
        self.t_min = np.inf

        # Sample a start time
        t = npr.rand() * T
        self._params = (t,)

    @property
    def posterior_mode(self):
        if np.isfinite(self.t_min):
            return self.t_min
        else:
            return 0.0

    def log_prior(self):
        return -np.log(self.T)

    def log_likelihood(self, data):
        t = self.params[0]
        return expon.logpdf(data - t, scale=self.tau)

    def add_datapoint(self, x):
        super(ExponentialTimeCluster, self).add_datapoint(x)
        # i = np.searchsorted(self.ts, x)
        # i = i[0] if isinstance(i, np.ndarray) else i
        # self.ts = np.insert(self.ts, i, x)
        if x < self.t_min:
            self.t_min = x

        self._params = (self.t_min - npr.exponential(scale=self.tau / self.size),)

    def remove_datapoint(self, x):
        super(ExponentialTimeCluster, self).remove_datapoint(x)
        # i = np.searchsorted(self.ts, x)
        # i = i[0] if isinstance(i, np.ndarray) else i
        # self.ts = np.delete(self.ts, i)
        if x == self.t_min:
            raise Exception("Removing the first event.  Recompute cluster!")

        self._params = (self.t_min - npr.exponential(scale=self.tau / self.size),)

    def sample(self, size=1):
        # Sample start time then exponentially distributed children
        s = npr.rand() * self.T
        ts = s + npr.exponential(scale=self.tau, size=size)
        # ts = np.clip(ts, 0, self.T)
        return ts

    def jitter(self, t_std=2.0):
        t = self.params[0]
        new_t = t + t_std * npr.randn()
        fwd_lp = rev_lp = 0
        new_cluster = copy.deepcopy(self)
        new_cluster._params = (new_t,)
        return new_cluster, fwd_lp, rev_lp

    def predictive_log_likelihood(self, t):
        N, t_min, tau = self.size, self.t_min, self.tau
        if N == 0:
            return -np.log(self.T)
        elif t > self.t_min:
            # N/(N+1) Exp(t' - t_min | tau)   if t' > t_min
            # return np.log(N) - np.log(N+1) + expon.logpdf(t - t_min, 0, tau)
            return np.log(N) - np.log(N+1) -(t - t_min) / tau - np.log(tau)
        else:
            # 1/(N+1) Exp(t_min - t' | tau/N) if t' < t_min
            # return - np.log(N+1) + expon.logpdf(t_min - t, 0, tau / N)
            return - np.log(N+1) -(t_min - t) * N / tau - np.log(tau) + np.log(N)
        return lp


class NodeIndexCluster(Cluster):
    """
    Cluster with categorical observations and a Dirichlet prior.

    The Dirichlet(a) prior density is

        p(u | a) = Gam(sum_k ak) / prod_k Gam(ak) * prod_k uk^{ak-1}

    The sufficient statistics are log(uk) for k = 1, .., K.

    The categorical likelihood is,

        p(x | u) = prod_k uk^I[x = k]

    So that the posterior is

        p(u | x, a) = Dir(a1 + x1, ..., aK + xK)

    The posterior predictive is
        p(x' = j| a') = Gam(2) Gam(sum_k a'_k) / Gam(1 + sum_k ak)
                       * prod_k Gam(I[k = j] + a_k) / Gam(I[k = j] + 1) / Gam(a_k)

                      = 1 / (sum_k ak) * Gam(a_j + 1) / Gam(a_j)
                      = a_j / sum_k a_k
    """
    def __init__(self, num_nodes=1, concentration=None):
        super(NodeIndexCluster, self).__init__()

        # Validate hyperparameters
        assert isinstance(num_nodes, int) and num_nodes >= 1
        assert (np.isscalar(concentration) and concentration > 0) \
            or (concentration.shape == (num_nodes,) and np.all(concentration > 0))
        self.num_nodes = num_nodes
        self.concentration = concentration

        # Store the natural parameters of the posterior
        self.a = concentration * np.ones(num_nodes)

        # Sample parameters
        pi = npr.dirichlet(self.a)
        self._params = (pi,)

    @property
    def posterior_mode(self):
        return self.a / self.a.sum()

    def sample_posterior(self):
        pi = npr.dirichlet(self.a)
        self._params = (pi,)

    def log_prior(self):
        return dirichlet(self.a).logpdf(self.params[0])

    def log_likelihood(self, data):
        log_pi = np.log(self.params[0])
        return log_pi[data]

    def add_datapoint(self, x):
        super(NodeIndexCluster, self).add_datapoint(x)
        self.a[x] += 1
        self.sample_posterior()

    def remove_datapoint(self, x):
        super(NodeIndexCluster, self).remove_datapoint(x)
        self.a[x] -= 1
        self.sample_posterior()

    def sample(self, size=1):
        # Sample Dirichlet then multinomial
        pi = npr.dirichlet(self.a)
        return npr.choice(self.num_nodes, p=pi, size=size)

    def jitter(self, concentration=100):
        pi = self.params[0]
        new_pi = npr.dirichlet(concentration * pi) + 1e-8
        new_pi /= new_pi.sum()
        fwd_lp = dirichlet(concentration * pi).logpdf(new_pi)
        rev_lp = dirichlet(concentration * new_pi).logpdf(pi)

        new_cluster = copy.deepcopy(self)
        new_cluster._params = (new_pi,)
        return new_cluster, fwd_lp, rev_lp

    def predictive_log_likelihood(self, x):
        """
        p(x' = j | {x_n}) = a_j / sum_k a_k
        """
        return np.log(self.a[x]) - np.log(self.a.sum())


class TimeAndMarkCluster(Cluster):
    """
    Combine a temporal model and a mark model
    """
    def __init__(self, time_class, mark_class, time_kwargs={}, mark_kwargs={}):
        super(TimeAndMarkCluster, self).__init__()

        self.time_cluster = time_class(**time_kwargs)
        self.mark_cluster = mark_class(**mark_kwargs)

    @property
    def params(self):
        return (self.time_cluster.params, self.mark_cluster.params)

    def log_prior(self):
        return self.time_cluster.log_prior() + \
               self.mark_cluster.log_prior()

    def log_likelihood(self, data):
        time, mark = data[:, 0], data[:, 1:]
        return self.time_cluster.log_likelihood(time) + \
               self.mark_cluster.log_likelihood(mark)

    def add_datapoint(self, x):
        super(TimeAndMarkCluster, self).add_datapoint(x)

        time, mark = x[0], x[1:]
        self.time_cluster.add_datapoint(time)
        self.mark_cluster.add_datapoint(mark)

    def remove_datapoint(self, x):
        super(TimeAndMarkCluster, self).remove_datapoint(x)

        time, mark = x[0], x[1:]
        self.time_cluster.remove_datapoint(time)
        self.mark_cluster.remove_datapoint(mark)

    def sample(self, size=1, time_kwargs={}, mark_kwargs={}):
        times = self.time_cluster.sample(size=size, **time_kwargs)
        marks = self.mark_cluster.sample(size=size, **mark_kwargs)
        return np.column_stack((times, marks))

    def jitter(self):
        new_time_cluster, fwd_lp1, rev_lp1 = self.time_cluster.jitter()
        new_mark_cluster, fwd_lp2, rev_lp2 = self.mark_cluster.jitter()

        new_cluster = copy.deepcopy(self)
        new_cluster.time_cluster = new_time_cluster
        new_cluster.mark_cluster = new_mark_cluster

        return new_cluster, fwd_lp1 + fwd_lp2, rev_lp1 + rev_lp2

    def predictive_log_likelihood(self, x):
        time, mark = x[0], x[1:]
        ll = self.time_cluster.predictive_log_likelihood(time)
        ll += self.mark_cluster.predictive_log_likelihood(mark)
        return ll


class NodeAndTimeAndMarkCluster(Cluster):
    """
    Combine a temporal model, a mark model, and a "node index" model
    The node index specifies which of the M nodes the event occurs on.
    """
    def __init__(self, num_nodes, time_class, mark_class,
                 node_concentration=1, time_kwargs={}, mark_kwargs={}):

        super(NodeAndTimeAndMarkCluster, self).__init__()
        self.num_nodes = num_nodes
        self.node_cluster = NodeIndexCluster(num_nodes=num_nodes, concentration=node_concentration)
        self.time_cluster = time_class(**time_kwargs)
        self.mark_cluster = mark_class(**mark_kwargs)

    @property
    def params(self):
        return (self.node_cluster.params,
                self.time_cluster.params,
                self.mark_cluster.params)

    def log_prior(self):
        return self.node_cluster.log_prior() + \
               self.time_cluster.log_prior() + \
               self.mark_cluster.log_prior()

    def log_likelihood(self, data):
        node = data[:, 0].astype(int)
        time = data[:, 1]
        mark = data[:, 2:]
        return self.node_cluster.log_likelihood(node) + \
               self.time_cluster.log_likelihood(time) + \
               self.mark_cluster.log_likelihood(mark)

    def add_datapoint(self, x):
        super(NodeAndTimeAndMarkCluster, self).add_datapoint(x)

        node = x[0].astype(int)
        time = x[1]
        mark = x[2:]
        self.node_cluster.add_datapoint(node)
        self.time_cluster.add_datapoint(time)
        self.mark_cluster.add_datapoint(mark)

    def remove_datapoint(self, x):
        super(NodeAndTimeAndMarkCluster, self).remove_datapoint(x)

        node = x[0].astype(int)
        time = x[1]
        mark = x[2:]
        self.node_cluster.remove_datapoint(node)
        self.time_cluster.remove_datapoint(time)
        self.mark_cluster.remove_datapoint(mark)

    def sample(self, size=1, time_kwargs={}, mark_kwargs={}):
        nodes = self.node_cluster.sample(size=size)
        times = self.time_cluster.sample(size=size, **time_kwargs)
        marks = self.mark_cluster.sample(size=size, **mark_kwargs)
        return np.column_stack((nodes, times, marks))

    def jitter(self):
        new_node_cluster, fwd_lp1, rev_lp1 = self.node_cluster.jitter()
        new_time_cluster, fwd_lp2, rev_lp2 = self.time_cluster.jitter()
        new_mark_cluster, fwd_lp3, rev_lp3 = self.mark_cluster.jitter()

        new_cluster = copy.deepcopy(self)
        new_cluster.node_cluster = new_node_cluster
        new_cluster.time_cluster = new_time_cluster
        new_cluster.mark_cluster = new_mark_cluster

        return new_cluster, fwd_lp1 + fwd_lp2 + fwd_lp3, rev_lp1 + rev_lp2 + rev_lp3

    def predictive_log_likelihood(self, x):
        node = x[0].astype(int)
        time = x[1]
        mark = x[2:]

        ll = self.node_cluster.predictive_log_likelihood(node)
        ll += self.time_cluster.predictive_log_likelihood(time)
        ll += self.mark_cluster.predictive_log_likelihood(mark)
        return ll


class SeqNMFCluster(Cluster):
    """
    In the sequence NMF cluster, each latent event has a discrete "motif type."
    Each motif type corresponds to a conditional distribution over node indices
    and, given the motif type and node index, a conditional distribution over
    the relative time of the child events.

    Thus, the latent events consist of a time (s in [0, T]) and a type (z in {1, ..., M}).
    They follow a simple prior,

        p(s) = Unif(s | [0, T])
        p(z) = Cat(z | eta)

    where eta is a distribution over the M motif types.

    The likelihood of a spike at time t on neuron n is

        p(t, n | s, z) = Cat(n | pi_z) N(t | s + Delta_{n,z}, sigma^2_{n, z})

    We get the predictive likelihood of a new event (t', n') by computing the
    log normalizers

    log p((t', n') | {(t_i, n_i)}) = log p((t', n') \cup {(t_i, n_i)}) - log p({(t_i, n_i)})

    where

    p({(t_i, n_i)}) = \sum_z \int p({(t_i, n_i)}, s, z) ds
                    = \sum_z \int p(s) p(z) \prod_i p(t_i | s, z) p(n_i | z) ds
                    = \sum_z p(z) \prod_i p(n_i | z) \int p(s) \prod_i p(t_i | s, z) ds

    The integral over s is a simple truncated Gaussian,

    \int p(s) \prod_i p(t_i | s, z) ds
        = \int 1/T I[s \in [0, T]] \prod_i N(t_i | s + \Delta_{n_i, z}, sigma^2_{n_i, z})
        = 1/T \int_0^T \prod_i N(s | t_i - \Delta_{n_i, z}, sigma^2_{n_i, z})
        = 1/T Z(t', sigma^2') \prod_i Z(t_i - \Delta_{n_i, z}, sigma^2_{n_i, z})
            \int_0^T N(s | t', sigma^2')

        = 1/T Z(t', sigma^2') \prod_i Z(t_i - \Delta_{n_i, z}, sigma^2_{n_i, z})
            * (Phi(T | t', sigma^2') - Phi(0 | t', sigma^2'))

    where

        Z(t, sigma^2) = exp{1/2 t^2 / sigma^2} * sqrt{2 * pi * sigma^2}

    and

        J' = \sum_i 1/sigma^2_{n_i, z}
        h' = \sum_i (t_i - Delta_{n_i, z}) / sigma^2_{n_i, z}
        t' = h'/J'
        sigma^2' = 1/J'

    """
    def __init__(self, T, num_motif_types, num_neurons, eta, pis, deltas, sigmasqs):

        super(SeqNMFCluster, self).__init__()
        self.T = T
        self.M = num_motif_types
        self.N = num_neurons

        # Check and store hyperparameters
        is_distribution = lambda p: np.all(p >= 0) and np.allclose(p.sum(axis=-1), 1.0)
        assert eta.shape == (self.M,) and is_distribution(eta)
        assert pis.shape == (self.M, self.N) and is_distribution(pis)
        assert deltas.shape == (self.M, self.N)
        assert sigmasqs.shape == (self.M, self.N) and np.all(sigmasqs > 0)

        self.eta = eta
        self.pis = pis
        self.deltas = deltas
        self.sigmasqs = sigmasqs

        # Initialize sufficient statistics of this cluster
        self.counts = np.zeros(self.N)
        self.Js = np.zeros(self.M)
        self.hs = np.zeros(self.M)
        self.log_Zs = np.zeros(self.M)

    def add_datapoint(self, x):
        super(SeqNMFCluster, self).add_datapoint(x)

        # Update sufficient statistics
        n, t = int(x[0]), x[1]
        self.counts[n] += 1
        self.Js += 1 / self.sigmasqs[:, n]
        self.hs += (t - self.deltas[:, n]) / self.sigmasqs[:, n]
        self.log_Zs += (0.5 * np.log(2 * np.pi * self.sigmasqs[:, n]) + \
                        0.5 * (t - self.deltas[:, n])**2 / self.sigmasqs[:, n])

    def remove_datapoint(self, x):
        super(SeqNMFCluster, self).remove_datapoint(x)

        # Update sufficient statistics
        n, t = int(x[0]), x[1]
        self.counts[n] -= 1
        self.Js -= 1 / self.sigmasqs[:, n]
        self.hs -= (t - self.deltas[:, n]) / self.sigmasqs[:, n]
        self.log_Zs -= (0.5 * np.log(2 * np.pi * self.sigmasqs[:, n]) + \
                        0.5 * (t - self.deltas[:, n])**2 / self.sigmasqs[:, n])

    def sample(self, size=1, time_kwargs={}, mark_kwargs={}):
        motif_type = npr.choice(self.M, p=self.eta)
        motif_time = npr.rand() * self.T
        neurons = npr.choice(self.N, p=self.pis[motif_type], size=size)
        spike_deltas = self.deltas[motif_type, neurons]
        spike_sigmasqs = self.sigmasqs[motif_type, neurons]
        spike_times = motif_time + spike_deltas + npr.rand(size) * np.sqrt(spike_sigmasqs)

        # Only keep valid spikes
        valid = (spike_times > 0) & (spike_times < self.T)
        return np.column_stack((neurons[valid], spike_times[valid]))

    def _compute_posterior_stats(self):
        assert self.size > 0

        # First compute the log normalizer for each value of z
        sigmasq_post = 1 / self.Js
        sigma_post = np.sqrt(sigmasq_post)
        mu_post = self.hs / self.Js
        log_Z_post = 0.5 * np.log(2 * np.pi * sigmasq_post) + 0.5 * mu_post ** 2 / sigmasq_post
        mll = -np.log(self.T) - self.log_Zs + log_Z_post

        # Note: This integral is nearly always 1, so its log is 0
        #       We will omit it since it's slow to compute
        #
        # Compute the Gaussian integral over [0, T]
        # for m in range(self.M):
        #     mll[m] += logsumexp([norm.logcdf(self.T, loc=mu_post[m], scale=sigma_post[m]),
        #                          norm.logcdf(0, loc=mu_post[m], scale=sigma_post[m])],
        #                          b=[1, -1])

        # Add the prior on motif types and the
        # likelihood of the spike counts for each motif type
        mll += np.log(self.eta)
        mll += np.sum(np.log(self.pis) * self.counts, axis=1)

        return mll, mu_post, sigma_post

    @property
    def posterior_mode(self):
        """
        Compute the most likely motif type and time
        """
        if self.size == 0:
            return np.argmax(self.eta), self.T // 2

        mll, mu_post, sigma_post = self._compute_posterior_stats()
        most_likely_type = np.argmax(mll)
        most_likely_time = mu_post[most_likely_type]
        return most_likely_type, most_likely_time

    def sample_posterior(self):
        """
        Compute the most likely motif type and time
        """
        if self.size == 0:
            return np.random.choice(self.M, p=self.eta), self.T * npr.rand()

        mll, mu_post, sigma_post = self._compute_posterior_stats()
        z = npr.choice(self.M, p=np.exp(mll - logsumexp(mll)))
        s = mu_post[z] + sigma_post[z] * npr.randn()

        return z, s

    def log_normalizer(self):
        if self.size == 0:
            return 0

        # Get the marginal log likelihood of each type and sum over z
        mll, _, _ = self._compute_posterior_stats()
        out = logsumexp(mll)
        assert np.isfinite(out)

        return out

    def predictive_log_likelihood(self, x):
        """
        Explicitly calculate the predictive log likelihood with
        all its normalizing constants.
        """
        n, t = int(x[0]), x[1]

        if self.size == 0:
            pll = logsumexp(np.log(self.eta) + np.log(self.pis[:, n])) - np.log(self.T)
        else:
            mll, mu_post, sigma_post = self._compute_posterior_stats()
            log_pz = mll - logsumexp(mll)

            # p(t | z) = \int p(t | s, z) p(s | z) ds
            #          = \int N(t | s + delta_z, sigma_z^2) N(s | mu_post, sigma_post^2)
            #          = N(t | mu_post + delta_z, sigma_post^2 + sigma_z^2)
            log_pt_given_z = norm.logpdf(t, mu_post + self.deltas[:, n], np.sqrt(sigma_post**2 + self.sigmasqs[:, n]))
            log_pn_given_z = np.log(self.pis[:, n])

            # p(t, n) = \sum_z p(t | z) p(n | z) p(z)
            pll = logsumexp(log_pz + log_pt_given_z + log_pn_given_z)

            # DEBUG
            # old_pll = super(SeqNMFCluster, self).predictive_log_likelihood(x)
            # assert np.allclose(pll, old_pll)

        return pll


class SeqNMFClusterFactory(ClusterFactory):
    """
    By default, treat the cluster hyperparameters as fixed.
    """
    def __init__(self, T, num_motif_types, num_neurons,
                 eta_conc, pi_conc,
                 delta_mean, delta_nu,
                 sigmasq_a, sigmasq_b):

        self.T = T
        self.M = num_motif_types
        self.N = num_neurons

        # Store the hyperparameters
        self.eta_conc = eta_conc
        self.pi_conc = pi_conc
        self.delta_mean = delta_mean
        self.delta_nu = delta_nu
        self.sigmasq_a = sigmasq_a
        self.sigmasq_b = sigmasq_b

        # Initialize the parameters
        self.eta = npr.dirichlet(eta_conc * np.ones(self.M))
        self.pis = npr.dirichlet(pi_conc * np.ones(self.N), size=self.M)
        self.sigmasqs = 1 / npr.gamma(sigmasq_a, 1 / sigmasq_b, size=(self.M, self.N))
        self.deltas = delta_mean + np.sqrt(self.sigmasqs / delta_nu) * npr.randn(self.M, self.N)

    def make_new_cluster(self):
        return SeqNMFCluster(self.T, self.M, self.N, self.eta, self.pis, self.deltas, self.sigmasqs)

    def gibbs_step(self, clusters, data, parents):
        # Unpack the data
        neuron = data[:, 0].astype(int)
        time = data[:, 1]

        # Sample the cluster types and times
        K = len(clusters)
        cluster_params = [cluster.sample_posterior() for cluster in clusters]
        cluster_types, cluster_times = list(zip(*cluster_params))

        # First update the posterior on the cluster types
        eta_conc = self.eta_conc * np.ones(self.M) + np.bincount(cluster_types, minlength=self.M)
        self.eta = npr.dirichlet(eta_conc)

        # Now update pi, delta, sigmasq for each cluster type
        for m in range(self.M):
            # update pi
            pi_conc = self.pi_conc * np.ones(self.N)
            for k in range(K):
                if cluster_types[k] == m:
                    pi_conc += np.bincount(neuron[parents == k], minlength=self.N)
            self.pis[m] = npr.dirichlet(pi_conc)

            # update delta and sigmasq from the NIG posterior
            spk_count = np.zeros(self.N)
            spk_delta_sum = np.zeros(self.N)
            spk_delta_sumsq = np.zeros(self.N)
            for k in range(K):
                if cluster_types[k] == m:
                    spk_count += np.bincount(neuron[parents == k], minlength=self.N)
                    for n in range(self.N):
                        spk_delta = time[(parents == k) & (neuron == n)] - cluster_times[k]
                        spk_delta_sum[n] += np.sum(spk_delta)
                        spk_delta_sumsq[n] += np.sum(spk_delta**2)

            spk_delta_mean = spk_delta_sum / spk_count
            spk_delta_mean[spk_count == 0] = 0
            # \sum_n (x_n - xbar)^2 = [sum (xn^2 - 2 xbar * x_n + xbar^2)]
            #                       = [sum xn^2 - 2 xbar * sum xn + n * xbar^2]
            #                       = [sum xn^2 - n * xbar^2]
            #                       = [sum xn^2 - 1/n * (sum xn) (sum xn)]
            a_post = self.sigmasq_a + spk_count / 2
            b_post = self.sigmasq_b + 0.5 *  (spk_delta_sumsq - spk_count * spk_delta_mean**2) \
                        + spk_count * self.delta_nu / (spk_count + self.delta_nu) * (spk_delta_mean - self.delta_mean)**2 / 2
            nu_post = self.delta_nu + spk_count
            mu_post = (self.delta_nu * self.delta_mean + spk_delta_sum) / (self.delta_nu + spk_count)

            # update sigmasq and delta
            self.sigmasqs[m] = 1 / npr.gamma(a_post, 1 / b_post)
            self.deltas[m] = mu_post + np.sqrt(self.sigmasqs[m] / nu_post) * npr.randn(self.N)


    def m_step(self, data, parents):
        raise NotImplementedError
