import copy
import numpy as np
import numpy.random as npr
from scipy.special import logsumexp, gammaln
from scipy.stats import poisson, gamma
from tqdm.auto import trange

from neymanscott.util import convex_combo

class FiniteMixtureModel(object):
    """
    Standard mixture model with a fixed, finite number of clusters.
    """
    def __init__(self,
                 alpha,
                 num_clusters,
                 observation_class,
                 observation_hypers=None):

        self.num_clusters = num_clusters

        # Initialize the prior
        self.alpha = alpha
        self.weights = npr.dirichlet(alpha * np.ones(self.num_clusters))

        # Initialize the observations
        self.observation_hypers = observation_hypers
        self.observation_class = observation_class

    def initialize_clusters(self, data, method="prior", z0=None):

        # Initialize the number of clusters
        if method == "given":
            assert z0 is not None
            assert z0.size == data.shape[0] and z0.dtype == int
            assert np.all(z0 >= 0 and z0 < self.num_clusters)
            parents = z0.copy()

        else:
            if method == "prior":
                parents = npr.choice(self.num_clusters, p=self.weights, size=data.shape[0])
            elif method == "single_cluster":
                parents = np.zeros(data.shape[0], dtype=int)
            else:
                raise Exception("invalid initialization method: {}".format(method))

        # Initialize the clusters
        clusters = [self.observation_class(**self.observation_hypers)
                    for _ in range(self.num_clusters)]
        for x, z in zip(data, parents):
            clusters[z].add_datapoint(x)

        return clusters, parents

    def generate(self, size, **kwargs):
        # Sample number of datapoints per cluster
        cluster_sizes = npr.multinomial(size, self.prior)

        # Sample data from each cluster
        data, parents = [], []
        for i, n in enumerate(cluster_sizes):
            cluster = self.observation_class(**self.observation_hypers)
            data.append(cluster.sample(size=n, **kwargs))
            parents.append(i * np.ones(n, dtype=int))

        data = np.concatenate(data, axis=0)
        parents = np.concatenate(parents)

        # Permute the data
        perm = npr.permutation(size)
        data = data[perm]
        parents = parents[perm]

        return data, parents

    def uncollapsed_gibbs_sample_posterior(self, data, num_samples=100, init_method="prior", z0=None):
        """
        Gibbs sample the posterior distribution.
        Note: this sampler is _collapsed_ over the cluster parameters
              but _uncollapsed_ over the mixture weights.
        """

        # Initialize the clusters randomly
        clusters, parents = self.initialize_clusters(data, method=init_method, z0=z0)

        # Run the Gibbs sampler
        copy_state = lambda: dict(parents=parents.copy(),
                                  prior=self.prior.copy(),
                                  clusters=[c.posterior_mode for c in clusters])
        samples = [copy_state()]
        for i in trange(num_iters):
            # Sample new parent assignments
            for n, x in enumerate(data):
                # Remove the datapoint from its current cluster
                clusters[parents[n]].remove_datapoint(x)

                # Evaluate the predictive log likelihood under each cluster
                pll = np.log(self.weights)
                for k, cluster in enumerate(clusters):
                    pll[k] += cluster.predictive_log_likelihood(x)

                # Sample a new cluster according to predictive log likelihoods
                pll -= logsumexp(pll)
                parents[n] = npr.choice(self.num_clusters, p=np.exp(pll))
                clusters[parents[n]].add_datapoint(x)

            # Sample new prior weights
            cluster_sizes = np.bincount(parents, minlength=self.num_clusters)
            self.weights = npr.dirichlet(self.alpha + cluster_sizes)

            # Store sample
            samples.append(copy_state())

        return samples

    def gibbs_sample_posterior(self, data, num_samples=100, init_method="prior", z0=None):
        """
        Gibbs sample the posterior distribution.
        Note: this sampler is _collapsed_ over both the cluster parameters
              and the mixture weights. This should look a lot like the DPMM sampler.
        """

        # Initialize the clusters randomly
        clusters, parents = self.initialize_clusters(data, method=init_method, z0=z0)

        # Run the Gibbs sampler
        copy_state = lambda: dict(parents=parents.copy(),
                                  prior=self.prior.copy(),
                                  clusters=[c.posterior_mode for c in clusters])
        samples = [copy_state()]
        for i in trange(num_iters):
            # Sample new parent assignments
            for n, x in enumerate(data):
                # Remove the datapoint from its current cluster
                clusters[parents[n]].remove_datapoint(x)

                # Evaluate the predictive log likelihood under each cluster
                pll = np.zeros(self.num_clusters)
                for k, cluster in enumerate(clusters):
                    pll[k] = np.log(cluster.size + self.alpha)
                    pll[k] += cluster.predictive_log_likelihood(x)

                # Sample a new cluster according to predictive log likelihoods
                pll -= logsumexp(pll)
                parents[n] = npr.choice(self.num_clusters, p=np.exp(pll))
                clusters[parents[n]].add_datapoint(x)

            # Store sample
            samples.append(copy_state())

        return samples


class DirichletProcessMixtureModel(object):
    def __init__(self,
                 alpha,
                 cluster_factory,
                 a0=1, b0=1):
        """
        alpha: concentration of the DP
        observation_class: class for cluster observations
        observation_hypers: hyperparameters of cluster observation class
        a0 and b0: hyperparameters of gamma prior on alpha
        """

        # Initialize the prior
        self.alpha = alpha
        self.a0 = a0
        self.b0 = b0
        assert alpha > 0 and a0 > 0 and b0 > 0

        # Initialize the observations
        self.cluster_factory = cluster_factory

    def initialize_clusters(self, data, method="single_cluster", z0=None):

        # Initialize the number of clusters
        if method == "given":
            assert z0 is not None
            assert z0.size == data.shape[0] and z0.dtype == int
            parents = z0.copy()
            num_clusters = len(np.unique(z0[z0 > -1]))
            assert np.all(sorted(np.unique(z0[z0 > -1])) == np.arange(num_clusters))

        else:
            if method == "single_cluster":
                num_clusters = 1
            elif method == "prior":
                num_clusters = max(int(self.alpha * np.log((self.alpha + len(data)) / self.alpha)), 1)
            else:
                raise Exception("invalid initialization method: {}".format(method))

            # Add data to clusters randomly
            prior = npr.dirichlet(self.alpha * np.ones(num_clusters))
            parents = npr.choice(num_clusters, p=prior, size=data.shape[0])

        # Initialize the clusters
        clusters = [self.cluster_factory.make_new_cluster()
                    for _ in range(num_clusters)]
        for x, z in zip(data, parents):
            clusters[z].add_datapoint(x)

        # Remove unused clusters and relabel
        cluster_sizes = np.bincount(parents[parents > -1], minlength=num_clusters)
        valid = np.where(cluster_sizes > 0)[0]
        clusters = [c for i, c in enumerate(clusters) if i in valid]
        new_parents = parents.copy()
        for newi, i in enumerate(valid):
            new_parents[parents == i] = newi
        assert np.all(np.diff(np.unique(new_parents)) == 1)
        assert np.all(np.array([c.size for c in clusters]) > 0)

        return clusters, new_parents

    def generate(self, size, **kwargs):
        num_clusters = npr.poisson(self.alpha * np.log((self.alpha + size) / self.alpha))
        assert num_clusters > 0

        # Sample number of datapoints per cluster
        prior = npr.dirichlet(self.alpha * np.ones(num_clusters))
        cluster_sizes = npr.multinomial(size, prior)

        # Sample data from each cluster
        data, parents, clusters = [], [], []
        for i, n in enumerate(cluster_sizes):
            cluster = self.cluster_factory.make_new_cluster()
            clusters.append(cluster)
            data.append(cluster.sample(size=n, **kwargs))
            parents.append(i * np.ones(n, dtype=int))

        data = np.concatenate(data, axis=0)
        parents = np.concatenate(parents)

        # Permute the data
        perm = npr.permutation(size)
        data = data[perm]
        parents = parents[perm]

        return data, parents, clusters

    def _gibbs_sample_alpha(self, num_data, num_clusters, num_steps=1):
        """
        Assuming alpha has a Ga(a, b) prior, follow the beta augmentation
        strategy of Escobar & West (1995).

        alpha | k, N, eta ~ pi_eta Ga(a0 + k, b0 - log eta)
                            + (1-pi_eta) Ga(a0 + k - 1, b0 - log eta)

        where pi_eta = (a0 + k - 1) / (n * (b0 - log eta))

        eta | alpha, k ~ Beta(alpha + 1, n)
        """
        eta = npr.beta(self.alpha + 1, num_data)
        component = npr.rand() < (self.a0 + num_clusters - 1) / \
                                    (num_data * (self.b0 - np.log(eta)))
        if component == 0:
            self.alpha = npr.gamma(self.a0 + num_clusters, 1 / (self.b0 - np.log(eta)))
        else:
            self.alpha = npr.gamma(self.a0 + num_clusters - 1, 1 / (self.b0 - np.log(eta)))


    def gibbs_sample_posterior(self, data, num_samples=100, init_method="prior", z0=None,
                               sample_alpha=True):
        # Number of datapoints
        N = data.shape[0]

        # Initialize the clusters randomly
        clusters, parents = self.initialize_clusters(data, method=init_method, z0=z0)
        num_clusters = len(clusters)

        # Run the Gibbs sampler
        copy_state = lambda: dict(parents=parents.copy(),
                                  num_clusters=num_clusters,
                                  clusters=[c.posterior_mode for c in clusters],
                                  alpha=self.alpha)
        samples = [copy_state()]
        for i in trange(num_samples):
            # Sample new parent assignments
            for n, x in enumerate(data):
                # Remove the datapoint from its current cluster
                curr_cluster = parents[n]
                # clusters[curr_cluster].remove_datapoint(x)
                try:
                    clusters[curr_cluster].remove_datapoint(x)
                except:
                    # Caught an exception when removing the datapoint.
                    # This is intentional with the ExponentialTimeCluster.
                    # Recreate the cluster from scratch with all but this datapoint.
                    repl_cluster = self.cluster_factory.make_new_cluster()
                    for m, xx in enumerate(data):
                        if m != n and parents[m] == curr_cluster:
                            repl_cluster.add_datapoint(xx)
                    clusters[curr_cluster] = repl_cluster

                # If this was the last datapoint in the cluster, remove it
                if clusters[curr_cluster].size == 0:
                    clusters = clusters[:curr_cluster] + clusters[curr_cluster+1:]
                    parents[parents > curr_cluster] -= 1
                    num_clusters -= 1

                # Evaluate the predictive log likelihood under each cluster
                pll = np.zeros(num_clusters + 1)
                for k, cluster in enumerate(clusters):
                    pll[k] = np.log(cluster.size) - np.log(N - 1 + self.alpha)
                    pll[k] += cluster.predictive_log_likelihood(x)

                # Evaluate predictivie likelihood under a new cluster
                new_cluster = self.cluster_factory.make_new_cluster()
                pll[-1] = np.log(self.alpha) - np.log(N - 1 + self.alpha)
                pll[-1] += new_cluster.predictive_log_likelihood(x)

                # Sample a new cluster according to predictive log likelihoods
                pll -= logsumexp(pll)
                parents[n] = npr.choice(num_clusters + 1, p=np.exp(pll))

                # If we sampled a new cluster, append it to our cluster list
                if parents[n] == num_clusters:
                    clusters.append(new_cluster)
                    num_clusters += 1

                # Add datapoint to its new cluster
                clusters[parents[n]].add_datapoint(x)

            # Sample the concentration parameter, if specified
            if sample_alpha:
                self._gibbs_sample_alpha(N, num_clusters)

            # Store sample
            samples.append(copy_state())

        return samples


class MixtureOfFiniteMixtureModel(object):
    """
    For starters, assume that the distribution over number of mixture
    components is Poisson with rate lambda.
    """
    def __init__(self,
                 mu,
                 alpha,
                 cluster_factory):

        """
        mu: Poisson rate for the prior on number of clusters
        alpha: Dirichlet concentration for weights on each cluster
        """
        self.mu = mu
        self.alpha = alpha
        self.cluster_factory = cluster_factory

    def precompute_log_Vs(self, n, threshold=1e-12):
        """
        Precompute the coefficients V_n(t) for each number of parts t.

        From Miller and Harrison,

            V_n(t) = \sum_{k=1}^\infty k_{(t)} / (alpha k)^{(n)} p_K(k)

        where

            x^{(m)} = x (x + 1) (x + 2) ... (x + m - 1)
                    = Gam(x+m) / Gam(x)
            x_{(m)} = x (x - 1) (x - 2) ... (x - m + 1)
                    = Gam(x+1) / Gam(x-m+1)

        with

            x_{(m)} = 0 if x is positive integer < m

        and

            x^{(0)} = 1 and x_{(0)} = 1.

        Truncated Poisson:

            Pr(x = k) = Po(x = k | lambda) / (1 - Po(x=0 | lambda))
                      = Po(x = k | lambda) / (1 - e^{-lambda})

        """
        log_V = -np.inf * np.ones(n)
        for t in trange(1, n+1):
            # Stop if:
            #   1. the next term in the sum is smaller than tolerance
            #   2. the prior CDF is greater than 1 - tolerance
            is_converged = False
            prior_cdf = 0
            k = t
            while not is_converged:
                trm1 = gammaln(k+1) - gammaln(k-t+1) - gammaln(self.alpha * k + n) + gammaln(self.alpha * k)
                trm2 = poisson(self.mu).logpmf(k) - np.log(1 - np.exp(-self.mu))
                prior_cdf += np.exp(trm2)

                # Update log V_n[t] with next term in sum
                tmp = logsumexp([trm1 + trm2, log_V[t-1]])
                delta = tmp - log_V[t-1]
                log_V[t-1] = tmp

                # DEBUG
                # if (k - t) % 100 == 0:
                #     print("n = ", n, "t = ", t, "k = ", k)

                # Check convergence
                if abs(delta) < threshold or prior_cdf > 1 - threshold:
                    is_converged = True
                    # print("converged after ", k-t, "iterations")

                # Proceed to next term
                k += 1

        return log_V

    def precompute_logp_k_given_t(self, log_Vs, maxk=100):
        """
        p(k | t) = 1/V_n(t) * k_{(t)} / (alpha k)^{(n)} p_K(k)
        """
        N = log_Vs.size
        logp = np.zeros((N, maxk))
        for t in range(1, N+1):
            assert t > 0
            for k in range(1, maxk+1):
                logp[t-1, k-1] = -log_Vs[t-1]
                logp[t-1, k-1] += gammaln(k+1) - gammaln(k-t+1)
                logp[t-1, k-1] += -gammaln(self.alpha * k + N) + gammaln(self.alpha * k)
                logp[t-1, k-1] += poisson(self.mu).logpmf(k) - np.log(1 - np.exp(-self.mu))
        return logp

    def initialize_clusters(self, data, method="single_cluster", z0=None):
        # Initialize the number of clusters
        if method == "given":
            assert z0 is not None
            assert z0.size == data.shape[0] and z0.dtype == int
            parents = z0.copy()
            num_clusters = len(np.unique(z0[z0 > -1]))
            assert np.all(sorted(np.unique(z0[z0 > -1])) == np.arange(num_clusters))

        else:
            if method == "single_cluster":
                num_clusters = 1
            elif method == "prior":
                num_clusters = min(1 + npr.poisson(self.mu), data.shape[0])
            else:
                raise Exception("invalid initialization method: {}".format(method))

            # Add data to clusters randomly
            prior = npr.dirichlet(self.alpha * np.ones(num_clusters))
            parents = npr.choice(num_clusters, p=prior, size=data.shape[0])

        # Initialize the clusters
        clusters = [self.cluster_factory.make_new_cluster()
                    for _ in range(num_clusters)]
        for x, z in zip(data, parents):
            clusters[z].add_datapoint(x)

        # Remove unused clusters and relabel
        cluster_sizes = np.bincount(parents[parents > -1], minlength=num_clusters)
        valid = np.where(cluster_sizes > 0)[0]
        clusters = [c for i, c in enumerate(clusters) if i in valid]
        new_parents = parents.copy()
        for newi, i in enumerate(valid):
            new_parents[parents == i] = newi
        assert np.all(np.diff(np.unique(new_parents)) == 1)
        assert np.all(np.array([c.size for c in clusters]) > 0)

        return clusters, new_parents

    def generate(self, size, **kwargs):
        num_clusters = min(1 + npr.poisson(self.mu), size)

        # Sample number of datapoints per cluster
        prior = npr.dirichlet(self.alpha * np.ones(num_clusters))
        cluster_sizes = npr.multinomial(size, prior)

        # Sample data from each cluster
        data, parents, clusters = [], [], []
        for i, n in enumerate(cluster_sizes):
            cluster = self.cluster_factory.make_new_cluster()
            clusters.append(cluster)
            data.append(cluster.sample(size=n, **kwargs))
            parents.append(i * np.ones(n, dtype=int))

        data = np.concatenate(data, axis=0)
        parents = np.concatenate(parents)

        # Permute the data
        perm = npr.permutation(size)
        data = data[perm]
        parents = parents[perm]

        return data, parents, clusters

    def gibbs_sample_posterior(self, data, num_samples=100, init_method="prior", z0=None,
                               geweke=False):
        # Number of datapoints
        N = data.shape[0]
        log_Vs = self.precompute_log_Vs(N)

        # Initialize the clusters randomly
        clusters, parents = self.initialize_clusters(data, method=init_method, z0=z0)
        num_clusters = len(clusters)

        # Run the Gibbs sampler
        copy_state = lambda: dict(parents=parents.copy(),
                                  num_clusters=num_clusters,
                                  clusters=[c.posterior_mode for c in clusters])
        samples = [copy_state()]
        for i in trange(num_samples):
            # Sample new parent assignments
            for n, x in enumerate(data):
                # Remove the datapoint from its current cluster
                curr_cluster = parents[n]
                # clusters[curr_cluster].remove_datapoint(x)
                try:
                    clusters[curr_cluster].remove_datapoint(x)
                except:
                    # Caught an exception when removing the datapoint.
                    # This is intentional with the ExponentialTimeCluster.
                    # Recreate the cluster from scratch with all but this datapoint.
                    repl_cluster = self.cluster_factory.make_new_cluster()
                    for m, xx in enumerate(data):
                        if m != n and parents[m] == curr_cluster:
                            repl_cluster.add_datapoint(xx)
                    clusters[curr_cluster] = repl_cluster

                # If this was the last datapoint in the cluster, remove it
                if clusters[curr_cluster].size == 0:
                    clusters = clusters[:curr_cluster] + clusters[curr_cluster+1:]
                    parents[parents > curr_cluster] -= 1
                    num_clusters -= 1

                # Evaluate the predictive log likelihood under each cluster
                pll = np.zeros(num_clusters + 1)
                for k, cluster in enumerate(clusters):
                    pll[k] = np.log(cluster.size + self.alpha)
                    pll[k] += cluster.predictive_log_likelihood(x)

                # Evaluate predictivie likelihood under a new cluster
                new_cluster = self.cluster_factory.make_new_cluster()
                pll[-1] = np.log(self.alpha)
                pll[-1] += log_Vs[num_clusters] - log_Vs[num_clusters - 1]
                pll[-1] += new_cluster.predictive_log_likelihood(x)

                # Sample a new cluster according to predictive log likelihoods
                pll -= logsumexp(pll)
                parents[n] = npr.choice(num_clusters + 1, p=np.exp(pll))

                # If we sampled a new cluster, append it to our cluster list
                if parents[n] == num_clusters:
                    clusters.append(new_cluster)
                    num_clusters += 1

                # Add datapoint to its new cluster
                clusters[parents[n]].add_datapoint(x)


            # Sample new data with the new clusters
            # if geweke:
            #     cluster_sizes = [c.size for c in clusters]
            #     data, parents, new_clusters = [], [], []
            #     for i, n in enumerate(cluster_sizes):
            #         new_cluster = self.cluster_factory.make_new_cluster()
            #         data.append(new_cluster.sample(size=n))
            #         parents.append(i * np.ones(n, dtype=int))
            #         new_clusters.append(new_cluster)

            #         # Add sampled datapoitns to the new cluster
            #         for x in data[-1]:
            #             new_cluster.add_datapoint(x)

            #     data = np.concatenate(data, axis=0)
            #     parents = np.concatenate(parents)
            #     clusters = new_clusters


            # Store sample
            samples.append(copy_state())

        return samples


class NeymanScottModel(object):
    """
    A latent point process model with point process observations
    and a single observed process.  Each latent event induces a
    random number of observed events.  Infer the number and locations
    of the latent events.
    """
    def __init__(self, mu, alpha, beta, lambda0, background, cluster_factory):

        """
        mu: Expected number of latent events
        alpha: Gamma shape for rates of each latent event
        beta: Gamma rate for rates of each latent event
        lambda0: Expected number of background events

        observation_class: class for observation model
        observation_hypers: hyperparameters to pass to observation model constructor
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.lambda0 = lambda0
        self.background = background
        self.cluster_factory = cluster_factory

        # Cache log_Vs for different values of n
        self._log_Vs_cache = dict()

    def precompute_log_Vs(self, n, threshold=1e-12):
        """
        Precompute the coefficients V_n(t) for each number of parts t.

        From Miller and Harrison,

            V_n(t) = \sum_{k=1}^\infty k_{(t)} (beta / (1+beta))^{alpha * k}

        where

            x_{(m)} = x (x - 1) (x - 2) ... (x - m + 1)
                    = Gam(x+1) / Gam(x-m+1)

        with

            x_{(m)} = 0 if x is positive integer < m


        """
        if n in self._log_Vs_cache:
            return self._log_Vs_cache[n]

        print("precomputing log_Vs")
        log_V = -np.inf * np.ones(n+1)
        for t in trange(n+1):
            # Stop if:
            #   1. the next term in the sum is smaller than tolerance
            #   2. the prior CDF is greater than 1 - tolerance
            is_converged = False
            prior_cdf = 0
            k = t
            while not is_converged:
                trm1 = gammaln(k+1) - gammaln(k-t+1)
                trm1 += self.alpha * k * (np.log(self.beta) - np.log(1+self.beta))
                trm2 = poisson(self.mu).logpmf(k)
                prior_cdf += np.exp(trm2)

                # Update log V_n[t] with next term in sum
                tmp = logsumexp([trm1 + trm2, log_V[t]])
                delta = tmp - log_V[t]
                log_V[t] = tmp

                # DEBUG
                # if (k - t) % 1 == 0:
                #     print("n = ", n, "t = ", t, "k = ", k, "delta = ", delta)

                # Check convergence
                if abs(delta) < threshold or prior_cdf > 1 - threshold:
                    is_converged = True
                    # reason = "delta < thresh" if abs(delta) < threshold else "cdf = 1"
                    # print("converged after ", k-t, "iterations bc ", reason)

                # Proceed to next term
                k += 1

        # Cache the log_Vs
        self._log_Vs_cache[n] = log_V

        return log_V

    def precompute_logp_k_given_t(self, log_Vs, maxk=100):
        """
        p(k | t) = 1/V_n(t) * k_{(t)} / (alpha k)^{(n)} p_K(k | n)

        TODO: double check this...
        """
        raise NotImplementedError

        # N = log_Vs.size
        # logp = np.zeros((N, maxk))
        # for t in range(1, N+1):
        #     assert t > 0
        #     for k in range(1, maxk+1):
        #         logp[t-1, k-1] = -log_Vs[t-1]
        #         logp[t-1, k-1] += gammaln(k+1) - gammaln(k-t+1)
        #         logp[t-1, k-1] += -gammaln(self.alpha * k + N) + gammaln(self.alpha * k)
        #         logp[t-1, k-1] += poisson(self.lmbda).logpmf(k)
        # return logp

    def initialize_clusters(self, data, method="single_cluster", z0=None):
        # Initialize the number of clusters
        if method == "given":
            assert z0 is not None
            assert z0.size == data.shape[0] and z0.dtype == int
            parents = z0.copy()
            num_clusters = len(np.unique(z0[z0 > -1]))
            assert np.all(sorted(np.unique(z0[z0 > -1])) == np.arange(num_clusters))

        else:
            if method == "prior":
                num_clusters = min(npr.poisson(self.mu), data.shape[0])
            elif method == "background":
                num_clusters = 0
            elif method == "single_cluster":
                num_clusters = 1
            else:
                raise Exception("invalid initialization method: {}".format(method))

            # Sample random weights
            weights = npr.gamma(self.alpha, 1 / self.beta, size=num_clusters)

            # Add data to clusters randomly
            prior = np.concatenate(([self.lambda0], weights))
            prior /= prior.sum()
            parents = npr.choice(num_clusters + 1, p=prior, size=data.shape[0]) - 1

        # Initialize clusters
        clusters = [self.cluster_factory.make_new_cluster() for _ in range(num_clusters)]
        for x, z in zip(data, parents):
            if z > -1:
                clusters[z].add_datapoint(x)

        # Remove unused clusters and relabel
        cluster_sizes = np.bincount(parents[parents > -1], minlength=num_clusters)
        valid = np.where(cluster_sizes > 0)[0]
        clusters = [c for i, c in enumerate(clusters) if i in valid]
        new_parents = parents.copy()
        for newi, i in enumerate(valid):
            new_parents[parents == i] = newi
        assert np.all(np.diff(np.unique(new_parents)) == 1)
        assert np.all(np.array([c.size for c in clusters]) > 0)

        return clusters, new_parents

    def generate(self, **kwargs):
        """
        Sample the number of clusters and the number of datapoints
        """
        data, parents, clusters, weights = [], [], [], []

        # Sample the background data
        nbkgd = npr.poisson(self.lambda0)
        if nbkgd > 0:
            xbkgd = self.background.sample(size=nbkgd, **kwargs)
            data.append(xbkgd)
            parents.append(-1 * np.ones(nbkgd, dtype=int))

        # Sample events for each latent cluster
        num_clusters = npr.poisson(self.mu)
        for k in range(num_clusters):
            # Sample cluster weight
            wk = npr.gamma(self.alpha, 1 / self.beta)
            # Sample the number of children
            nk = max(1, npr.poisson(wk))
            # Sample the marks of children
            cluster = self.cluster_factory.make_new_cluster()
            xk = cluster.sample(size=nk, **kwargs)
            nk = len(xk)
            for x in xk:
                cluster.add_datapoint(x)

            # Append the samples
            data.append(xk)
            parents.append(k * np.ones(nk, dtype=int))
            clusters.append(cluster)
            weights.append(wk)

        # Combine
        data = np.concatenate(data, axis=0)
        parents = np.concatenate(parents)

        # Permute the data
        perm = npr.permutation(len(parents))
        data = data[perm]
        parents = parents[perm]

        return data, parents, clusters, weights

    def log_probability(self, data, clusters, weights):
        num_clusters = len(clusters)
        lp = -self.mu + num_clusters * np.log(self.mu)
        for cluster, weight in zip(clusters, weights):
            lp += gamma(self.alpha, scale=1/self.beta).logpdf(weight)
            lp += cluster.log_prior()

        lp -= self.lambda0
        lp -= np.sum(weights)

        # Compute instantaneous log rates
        lls = [self.background.log_likelihood(data) + np.log(self.lambda0)]
        for cluster, weight in zip(clusters, weights):
            lls.append(cluster.log_likelihood(data) + np.log(weight))
        lls = np.column_stack(lls)
        lp += logsumexp(lls, axis=-1).sum()
        return lp


    def gibbs_sample_posterior(self, data, num_samples=100, init_method="background", z0=None,
                               verbose=True):
        """
        Use convention that z[n] == -1 means n'th event attributed to background
        """
        # Number of datapoints
        N = data.shape[0]

        # Precompute scaling factors
        log_Vs = self.precompute_log_Vs(N)

        # Initialize the clusters randomly
        clusters, parents = self.initialize_clusters(data, method=init_method, z0=z0)
        assert np.all(np.array([c.size for c in clusters]) > 0)
        num_clusters = len(clusters)
        num_bkgd = np.sum(parents == -1)

        # Run the Gibbs sampler
        # Instantiate the cluster weights from their gamma conditional
        # w ~ Ga(alpha, beta); n ~ Po(w) -> w | n ~ Ga(alpha + n, beta + 1)
        weights = [npr.gamma(self.alpha + cluster.size, 1 / (self.beta + 1)) for cluster in clusters]
        log_joint = self.log_probability(data, clusters, weights)
        copy_state = lambda: dict(log_prob=log_joint,
                                  parents=parents.copy(),
                                  num_clusters=num_clusters,
                                  clusters=clusters,
                                  weights=weights,
                                  modes=[c.posterior_mode for c in clusters])

        samples = [copy_state()]

        if verbose: print("Gibbs sampling event parents")
        pbar = trange(num_samples) if verbose else range(num_samples)
        for i in pbar:
            # Sample new parent assignments
            for n, x in enumerate(data):
                # Remove the datapoint from its current cluster
                curr_cluster = parents[n]

                if curr_cluster == -1:
                    num_bkgd -= 1

                else:
                    try:
                        clusters[curr_cluster].remove_datapoint(x)
                    except:
                        # Caught an exception when removing the datapoint.
                        # This is intentional with the ExponentialTimeCluster.
                        # Recreate the cluster from scratch with all but this datapoint.
                        repl_cluster = self.cluster_factory.make_new_cluster()
                        for m, xx in enumerate(data):
                            if m != n and parents[m] == curr_cluster:
                                repl_cluster.add_datapoint(xx)
                        clusters[curr_cluster] = repl_cluster

                    # If this was the last datapoint in the cluster, remove it
                    if clusters[curr_cluster].size == 0:
                        clusters = clusters[:curr_cluster] + clusters[curr_cluster+1:]
                        parents[parents > curr_cluster] -= 1
                        num_clusters -= 1

                # Compute predictive log likelihods under each parent assignment
                # (background, existing clusters, new cluster)
                pll = np.zeros(num_clusters + 2)

                # Evaluate likelihood under background
                pll[0] = np.log(self.lambda0) + np.log(1 + self.beta)
                pll[0] += self.background.log_likelihood(x)

                # Evaluate the predictive log likelihood under each existing cluster
                for k, cluster in enumerate(clusters):
                    pll[k+1] = np.log(cluster.size + self.alpha)
                    pll[k+1] += cluster.predictive_log_likelihood(x)

                # Evaluate predictivie likelihood under a new cluster
                new_cluster = self.cluster_factory.make_new_cluster()
                pll[-1] = np.log(self.alpha)
                # log V_N(t+1) - log V_N(t) where t is the number of clusters,
                # not including the new cluster.  Note that log_Vs starts with t=0.
                pll[-1] += log_Vs[num_clusters+1] - log_Vs[num_clusters]
                pll[-1] += new_cluster.predictive_log_likelihood(x)

                # Sample a new cluster according to predictive log likelihoods
                pll -= logsumexp(pll)
                assert np.all(np.isfinite(pll))
                parents[n] = npr.choice(num_clusters + 2, p=np.exp(pll)) - 1

                # If we sampled a new cluster, append it to our cluster list
                if parents[n] == num_clusters:
                    clusters.append(new_cluster)
                    num_clusters += 1

                # Add datapoint to its new cluster
                if parents[n] > -1:
                    clusters[parents[n]].add_datapoint(x)

            # Instantiate the cluster weights from their gamma conditional
            # w ~ Ga(alpha, beta); n ~ Po(w) -> w | n ~ Ga(alpha + n, beta + 1)
            weights = [npr.gamma(self.alpha + cluster.size, 1 / (self.beta + 1)) for cluster in clusters]

            # Store sample
            log_joint = self.log_probability(data, clusters, weights)
            samples.append(copy_state())
            print("num clusters: ", len(clusters), " lp: ", log_joint)

        return samples

    def rjmcmc_sample_posterior(self, data, num_samples=100, verbose=False,
                                init_method="background", clusters=None, weights=None,
                                qs=(1/3, 1/3, 1/3), gamma_shape=10):
        """
        Reversible jump MCMC.  Iteratively propose to add/remove/jitter latent events.

        The MH acceptance probabilities are derived in Appendix C.
        """

        # Initialize the clusters randomly
        ADD = 0
        REMOVE = 1
        JITTER = 2

        if init_method == "background":
            num_clusters = 0
            clusters = []
            weights = []
        elif init_method == "prior":
            if clusters is None and weights is None:
                num_clusters = min(npr.poisson(self.mu), data.shape[0])
                weights = [npr.gamma(self.alpha, 1 / self.beta) for _ in range(num_clusters)]
                clusters = [self.cluster_factory.make_new_cluster() for _ in range(num_clusters)]
        elif init_method == "given":
            num_clusters = len(clusters)
            clusters = copy.deepcopy(clusters)
            weights = copy.deepcopy(weights)
        else:
            raise NotImplementedError

        # Run the Gibbs sampler
        log_joint = self.log_probability(data, clusters, weights)
        copy_state = lambda: dict(log_prob=log_joint,
                                  num_clusters=num_clusters,
                                  clusters=clusters,
                                  modes=[c.posterior_mode for c in clusters])

        samples = [copy_state()]

        # Initialize the log likelihoods
        lls = [self.background.log_likelihood(data) + np.log(self.lambda0)]
        for cluster, weight in zip(clusters, weights):
            lls.append(cluster.log_likelihood(data) + np.log(weight))
        lls = np.column_stack(lls)

        if verbose: print("Gibbs sampling event parents")
        pbar = trange(num_samples)
        for i in pbar:

            if i % 500 == 0:
                print("num clusters: ", num_clusters, " lp: ", log_joint)

            # Choose add/remove/jitter
            move = npr.choice(3, p=np.array(qs))
            if move == ADD:
                if verbose: print("Propose ADD")
                new_cluster = self.cluster_factory.make_new_cluster()
                new_weight = npr.gamma(self.alpha, 1 / self.beta)

                logp = np.log(qs[1]) - np.log(qs[0])  # q_rem / q_add
                logp += np.log(self.mu) - new_weight - np.log(num_clusters + 1)

                # Compute instantaneous log rates
                new_lls = new_cluster.log_likelihood(data) + np.log(new_weight)
                dlls = logsumexp(np.column_stack((lls, new_lls)), axis=-1) - logsumexp(lls, axis=-1)
                logp += np.sum(dlls)

                if np.log(npr.rand()) < logp:
                    clusters.append(new_cluster)
                    weights.append(new_weight)
                    lls = np.column_stack((lls, new_lls))
                    if verbose: print("Accept")


            elif move == REMOVE:
                if verbose: print("Propose REMOVE")
                if num_clusters > 0:
                    k = npr.choice(num_clusters)
                    logp = np.log(qs[0]) - np.log(qs[1])  # q_add / q_rem
                    logp += - np.log(self.mu) + weights[k] + np.log(num_clusters)

                    # Compute instantaneous log rates
                    notk = np.concatenate(([0], 1 + np.arange(k), 1 + np.arange(k+1, num_clusters)))
                    dlls = logsumexp(lls[:, notk], axis=-1) - logsumexp(lls, axis=-1)
                    logp += np.sum(dlls)

                    if np.log(npr.rand()) < logp:
                        clusters = clusters[:k] + clusters[k+1:]
                        weights = weights[:k] + weights[k+1:]
                        lls = lls[:, notk]
                        if verbose: print("Accept")



            elif move == JITTER:
                if verbose: print("Propose JITTER")
                if num_clusters > 0:
                    k = npr.choice(num_clusters)
                    new_cluster, fwd_lp, rev_lp = clusters[k].jitter()
                    new_weight = gamma(gamma_shape * weights[k], scale=1 / gamma_shape).rvs()
                    fwd_lp += gamma(gamma_shape * weights[k], scale=1 / gamma_shape).logpdf(new_weight)
                    rev_lp += gamma(gamma_shape * new_weight, scale=1 / gamma_shape).logpdf(weights[k])

                    logp = rev_lp - fwd_lp
                    logp += gamma(self.alpha, scale=1/self.beta).logpdf(new_weight)
                    logp -= gamma(self.alpha, scale=1/self.beta).logpdf(weights[k])
                    logp += new_cluster.log_prior()
                    logp -= clusters[k].log_prior()
                    logp += -new_weight
                    logp -= -weights[k]

                    # Compute instantaneous log rates
                    curr_lls = lls[:, k+1].copy()
                    lls[:, k+1] = new_cluster.log_likelihood(data) + np.log(new_weight)
                    logp += np.sum(logsumexp(lls, axis=-1))
                    lls[:, k+1] = curr_lls
                    logp -= np.sum(logsumexp(lls, axis=-1))

                    if np.log(npr.rand()) < logp:
                        clusters[k] = new_cluster
                        weights[k] = new_weight
                        lls[:, k+1] = new_lls
                        if verbose: print("Accept")

            # Only recompute the log joint every 20 iterations for speed
            if i % 100 == 0:
                log_joint = self.log_probability(data, clusters, weights)

            num_clusters = len(clusters)
            samples.append(copy_state())

        return samples

    def gibbs_mh_sample_posterior(self, data, num_samples=100, verbose=False,
                                  init_method="background", clusters=None, weights=None,
                                  num_mh_per_gibbs=10):
        """
        Reversible jump MCMC.  Iteratively propose to add/remove/jitter latent events.

        The MH acceptance probabilities are derived in Appendix C.
        """

        # Initialize the clusters randomly
        ADD = 0
        REMOVE = 1

        if init_method == "background":
            num_clusters = 0
            clusters = []
            weights = []
        elif init_method == "prior":
            if clusters is None and weights is None:
                num_clusters = min(npr.poisson(self.mu), data.shape[0])
                weights = [npr.gamma(self.alpha, 1 / self.beta) for _ in range(num_clusters)]
                clusters = [self.cluster_factory.make_new_cluster() for _ in range(num_clusters)]
        elif init_method == "given":
            num_clusters = len(clusters)
            clusters = copy.deepcopy(clusters)
            weights = copy.deepcopy(weights)
        else:
            raise NotImplementedError

        # Run the Gibbs sampler
        log_joint = self.log_probability(data, clusters, weights)
        copy_state = lambda: dict(log_prob=log_joint,
                                  num_clusters=num_clusters,
                                  clusters=clusters,
                                  modes=[c.posterior_mode for c in clusters])

        samples = [copy_state()]

        # Initialize the log likelihoods
        lls = [self.background.log_likelihood(data) + np.log(self.lambda0)]
        for cluster, weight in zip(clusters, weights):
            lls.append(cluster.log_likelihood(data) + np.log(weight))
        lls = np.column_stack(lls)

        if verbose: print("Gibbs sampling event parents")
        pbar = trange(num_samples)
        for i in pbar:

            if i % 10 == 0:
                print("num clusters: ", num_clusters, " lp: ", log_joint)

            # Initialize parents
            ps = np.exp(lls - logsumexp(lls, axis=1, keepdims=True))
            parents = [npr.choice(num_clusters + 1, p=p) - 1 for p in ps]

            # Resample existing cluster parameters given parents by recreating them
            # from scratch.  In doing so, we automatically sample their parameters
            clusters = [self.cluster_factory.make_new_cluster() for _ in range(num_clusters)]
            for z, x in zip(parents, data):
                if z >= 0:
                    clusters[z].add_datapoint(x)

            # Resample weights
            weights = [npr.gamma(self.alpha + cluster.size, 1 / (self.beta + 1)) for cluster in clusters]

            # Recreate the lls
            lls = [self.background.log_likelihood(data) + np.log(self.lambda0)]
            for cluster, weight in zip(clusters, weights):
                lls.append(cluster.log_likelihood(data) + np.log(weight))
            lls = np.column_stack(lls)

            # Choose add/remove/jitter
            for mh_step in range(num_mh_per_gibbs):
                move = npr.choice(2)
                if move == ADD:
                    if verbose: print("Propose ADD")
                    new_cluster = self.cluster_factory.make_new_cluster()
                    new_weight = npr.gamma(self.alpha, 1 / self.beta)

                    logp = np.log(self.mu) - new_weight - np.log(num_clusters + 1)

                    # Compute instantaneous log rates
                    new_lls = new_cluster.log_likelihood(data) + np.log(new_weight)
                    dlls = logsumexp(np.column_stack((lls, new_lls)), axis=-1) - logsumexp(lls, axis=-1)
                    logp += np.sum(dlls)

                    if np.log(npr.rand()) < logp:
                        clusters.append(new_cluster)
                        weights.append(new_weight)
                        lls = np.column_stack((lls, new_lls))
                        if verbose: print("Accept")

                elif move == REMOVE:
                    if verbose: print("Propose REMOVE")
                    if num_clusters > 0:
                        k = npr.choice(num_clusters)
                        logp = - np.log(self.mu) + weights[k] + np.log(num_clusters)

                        # Compute instantaneous log rates
                        notk = np.concatenate(([0], 1 + np.arange(k), 1 + np.arange(k+1, num_clusters)))
                        dlls = logsumexp(lls[:, notk], axis=-1) - logsumexp(lls, axis=-1)
                        logp += np.sum(dlls)

                        if np.log(npr.rand()) < logp:
                            clusters = clusters[:k] + clusters[k+1:]
                            weights = weights[:k] + weights[k+1:]
                            lls = lls[:, notk]
                            if verbose: print("Accept")

                num_clusters = len(clusters)

            # Recompute log joint after the sequence of MH updates
            if i % 10 == 0:
                log_joint = self.log_probability(data, clusters, weights)

            samples.append(copy_state())

        return samples

    def _fit_gibbs(self, data, num_samples=100, init_method="background", z0=None):
        """
        Fit the model (latent events and background parameters) with Gibbs sampling.
        """
        # Initialize the parents
        _, parents = self.initialize_clusters(data, method=init_method, z0=z0)

        samples = []
        for itr in trange(num_samples):
            # Sample the latent events
            this_sample = self.gibbs_sample_posterior(
                data, num_samples=1, init_method="given", z0=parents, verbose=False)[-1]
            parents = this_sample["parents"]
            clusters = this_sample["clusters"]

            # Treat mu, alpha, beta as fixed hyperparameters

            # Sample the cluster hyperparameters
            self.cluster_factory.gibbs_step(clusters, data, parents)

            # Sample the background parameters
            self.background.gibbs_step(data[parents == -1])
            this_sample["background"] = copy.deepcopy(self.background)

            # Sample the background rate under Ga(1,1) prior and single Poisson observation
            lambda0_a_post = 1e-4 + np.sum(parents == -1)
            lambda0_b_post = 1e-4 + 1
            self.lambda0 = npr.gamma(lambda0_a_post, 1 / lambda0_b_post)

            # Update the sample
            this_sample["lambda0"] = self.lambda0
            samples.append(this_sample)

        return samples


    def m_step(self, data, samples, step_size=1):
        """
        Update the global parameters of the model to maximize
        the expected complete data log probability, i.e a lower
        bound on the marginal likelihood of the data.
        """
        parents = samples[-1]["parents"]
        n_bkgd = np.sum(parents == -1)
        n_latent = samples[-1]["num_clusters"]

        # Update the rate of latent events
        mu = convex_combo(self.mu, n_latent + 1e-16, step_size)
        self.mu = mu
        # print("mu: ", self.mu)

        # Update the cluster hyperparameters
        self.cluster_factory.m_step(data, parents)

        # Update the background rate of latent events
        lambda0 = convex_combo(self.lambda0, n_bkgd + 1e-16, step_size)
        self.lambda0 = lambda0
        # print("lambda0:", self.lambda0)

        # Note:  We are not updating the shape and scale of the gamma
        # prior on latent event weights since these are fixed hyperparameters.

        # Invalidate the log Vs cache since it depends on mu, alpha, beta
        self._log_Vs_cache = dict()

        # Update the hyperparameters of the background model
        self.background.m_step(data[parents == -1], step_size=step_size)

    def _fit_mcem(self, data, num_iters=50, num_gibbs_samples=10, step_size=1, verbose=False):
        """
        Fit the model parameters with Monte Carlo EM.
        Run the Gibbs sampler to approximate the expected log likelihood
        with samples from the conditional distribution over clusters.
        Update the model parameters with a step in the direction
        of the gradient of the expected log likelihood.
        """
        for itr in trange(num_iters):
            # E step: sample the posterior distribution over clusters
            samples = self.gibbs_sample_posterior(data, num_samples=num_gibbs_samples, verbose=verbose)

            # M step: update global parameters of the model
            self.m_step(data, samples, step_size=step_size)

    def fit(self, data, method, initialize=True, **kwargs):

        if initialize:
            self.background.initialize(data)
            self.lambda0 = max(len(data) - self.mu * self.alpha / self.beta, 1)

        methods = dict(mcem=self._fit_mcem,
                       gibbs=self._fit_gibbs)
        assert method in methods.keys(), \
            "Invalid fitting method {}. Must be one of {}".format(method, methods.keys())

        return methods[method](data, **kwargs)
