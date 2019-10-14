import os
import numpy as np
import numpy.random as npr
from scipy import sparse
from datetime import date, timedelta

from neymanscott.background import NodeAndTimeAndMarkBackground, UniformTimeBackground, MultinomialBackground
from neymanscott.clusters import NodeAndTimeAndMarkCluster, ExponentialTimeCluster, MultinomialCluster
from neymanscott.models import NeymanScottModel


# IO
DATA_DIR = "."
OUT_DIR = "."

# Preprocessing
VOCAB_OFFSET = 100
VOCAB_SIZE = 10000
T_MIN = 1270
T_MAX = 1290
MIN_EVENT_LENGTH = 10
MIN_EVENTS_PER_NODE = 10
MAX_EVENTS_PER_NODE = np.inf

def load_data():
    """
    """
    unique_uid = []
    with open(os.path.join(DATA_DIR, 'unique_docid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())

    unique_wid = []
    with open(os.path.join(DATA_DIR, 'unique_words.txt'), 'r') as f:
        for line in f:
            unique_wid.append(line.strip())

    unique_eid = []
    with open(os.path.join(DATA_DIR, 'unique_entities.txt'), 'r') as f:
        for line in f:
            unique_eid.append(line.strip())

    n_users = len(unique_uid) # doc
    n_items = len(unique_wid) # word
    n_entities = len(unique_eid)

    # Make mapping from word ids to words
    id2term = dict((i, termid) for (i, termid) in enumerate(unique_wid))
    print("num docs: ", n_users)
    print("num words: ", n_items)
    print("num entities: ", n_entities)


    # metadata includes information about author and time of each document
    # meta.csv: doc.id    author.id    time.id
    meta = np.loadtxt(os.path.join(DATA_DIR, 'meta.tsv'), delimiter='\t')
    ms_all = meta[:,1].astype(int)     # the labelling of nodes start from 0
    ts_all = meta[:,2].astype(int)     # make sure no events occur at time 0
    # word counts are stored as a sparse matrix
    ys_all = sparse.load_npz(os.path.join(DATA_DIR, 'train_data.npz'))

    # Extract the vocabulary
    word_counts = np.array(np.sum(ys_all, axis=0))[0]
    word_perm = np.argsort(word_counts)[::-1]
    valid_word_idxs = word_perm[VOCAB_OFFSET:VOCAB_OFFSET+VOCAB_SIZE]
    ys_valid = ys_all[:, valid_word_idxs]

    # Make a mapping from y index to vocab
    yidx2term = dict((i, id2term[id]) for (i, id) in enumerate(valid_word_idxs))
    term2yidx = dict((id2term[id], i) for (i, id) in enumerate(valid_word_idxs))

    # Extract the time window
    in_window = (ts_all >= T_MIN) & (ts_all <= T_MAX)
    ts, ms, ys = ts_all[in_window], ms_all[in_window], ys_valid[in_window]

    # Extract events that meet length requirements
    event_lengths = np.array(np.sum(ys, axis=1)).ravel()
    is_long_enough = event_lengths >= MIN_EVENT_LENGTH
    ts, ms, ys = ts[is_long_enough], ms[is_long_enough], ys[is_long_enough]

    # Cull nodes that are too active or not active enough
    unique_ms, num_events_per_node = np.unique(ms, return_counts=True)
    valid_ms = (num_events_per_node >= MIN_EVENTS_PER_NODE) & (num_events_per_node < MAX_EVENTS_PER_NODE)
    num_events_per_node = num_events_per_node[valid_ms]
    on_valid_node = np.in1d(ms, unique_ms[valid_ms])
    ts, ms, ys = ts[on_valid_node], ms[on_valid_node], ys[on_valid_node]

    # finally, relabel the nodes to be contiguous
    unique_ms, ms = np.unique(ms, return_inverse=True)
    # make mapping from new ms to original entity ids
    # midx2entity = dict((i, eid) for (i, eid) in zip(ms, unique_ms)) # yw / not sure it is correct?

    midx2entity = dict((i, eid) for (i, eid) in zip(np.arange(len(unique_ms)), unique_ms))
    # Permute the events by time
    perm = np.argsort(ts)
    ts = ts[perm]
    ms = ms[perm]
    ys = ys[perm]

    return ts, ms, ys, yidx2term, term2yidx, midx2entity, valid_word_idxs


ts, ms, ys, yidx2term, term2yidx, midx2entity, valid_word_idxs = load_data()

# Extract some helper info about the docs
event_lengths = np.array(np.sum(ys, axis=1)).ravel()

# Extract constant
N = len(ts)                  # number of datapoints
T = T_MAX - T_MIN            # length of time window
D = ys.shape[1]              # dimensionality of marks
M = int(ms.max() + 1)        # number of entities

print("num datapoints: ", N)
print("num time bins:  ", T)
print("num vocab:      ", D)
print("num entities:   ", M)


events_per_entity = np.bincount(ms)
# plt.plot(np.cumsum(sorted(events_per_entity)[::-1]))
# plt.xlabel("Entity index (sorted)")
# plt.ylabel("Total number of events")


# In[11]:


# plt.figure(figsize=(12, 8))
# events_per_day = np.bincount(ts - T_min, minlength=T_max - T_min + 1)
# plt.plot(np.arange(T_min, T_max+1), events_per_day)
# plt.xlabel("Time (day)")
# plt.ylabel("Total number of events")


# In[12]:


# plt.figure(figsize=(12, 8))
# # num_short_docs = np.array([np.array(ys[ts==t].sum(axis=1) < 10).sum() for t in range(T_min, T_max+1)])
# num_short_docs = np.array([(event_lengths[ts==t] < 10).sum() for t in range(T_min, T_max+1)])
# plt.plot(np.arange(T_min, T_max+1), num_short_docs)
# plt.xlabel("Time (day)")
# plt.ylabel("Nuumber of docs of length < 10")
# plt.xlim(1000, 1500)


# In[13]:


# # Find a random short doc around in the window of 1250-1300
# idxs = np.where((ts == 1274) & (event_lengths < 10))[0]

# for idx in idxs[:10]:
#     print("doc ", idx, ": ", [yidx2term[v] for v in ys[idx].nonzero()[1]])
#     print("")


# In[14]:


# print("Top 100 words: ", [yidx2term[i] for i in range(100)])


# # In[15]:


# # sanity check of top words in the range June 24 to July 14, 1976
# for t in range(1270, 1290):
#     print("day", t)
#     topwordid = np.array(np.argsort(ys[ts==t].sum(axis=0)))[0][::-1][:50]
#     print([yidx2term[idx] for idx in topwordid])


# In[16]:


# How many documents have the term "bicentennial" over time
def keyword_frequency(keyword):
    keyword_idx = term2yidx[keyword.upper()]
    keyword_count = np.array([ys[ts==t, keyword_idx].toarray().sum() for t in range(T_MIN, T_MAX)])
    return keyword_count


# In[17]:


# How many documents have the term "bicentennial" over time
def keyword_doc_frequency(keyword):
    keyword_idx = term2yidx[keyword.upper()]
    keyword_doc_count = np.array([(ys[ts==t, keyword_idx].toarray() > 0).sum() for t in range(T_MIN, T_MAX)])
    return keyword_doc_count


# In[18]:


# plt.figure(figsize=(12, 8))

keywords = ["bicentennial", "Uganda", "Khartoum"]

# for i, kwd in enumerate(keywords):
#     plt.subplot(len(keywords), 1, i+1)
#     plt.plot(keyword_frequency(kwd), label=kwd, alpha=1)
#     plt.xlabel("time (days)")
#     plt.ylabel("total word count")
# #     plt.legend(loc="upper left")
#     plt.title(kwd)
# #     plt.xlim(1200, 1290)

# plt.subplot(212)
# plt.plot(keyword_doc_frequency("bicentennial"), label="bicentennial", alpha=0.75)
# plt.plot(keyword_doc_frequency("Uganda"), label="Uganda", alpha=0.75)
# plt.xlabel("time (days)")
# plt.ylabel("total docs appeared in")
# # plt.xlim(1200, 1290)


# In[19]:


# Jitter the timestamps a little bit to make them non-identical
ts_jit = ts + npr.rand(N)


# # Try to fit the cables model

# In[21]:



# In[22]:


# Extract a mini-batch of data
mb_start = 1270   # June 24, 1976
mb_end = 1290   # July 14, 1976
T_mb = mb_end - mb_start + 1
in_window = (ts_jit >= mb_start) & (ts_jit <= mb_end)
N_mb = np.sum(in_window)

ms_mb = ms[in_window]
ts_mb = ts_jit[in_window]
ys_mb = ys[in_window]
data = np.array(np.column_stack((ms_mb, ts_mb, ys_mb.todense())))


# In[23]:


# Make a Neyman-Scott model with multinomial marks
tau = 3.                # time constant of exponential impulse responses
mu = 0.1 * T_mb         # expected number of latent events
alpha = 4.0             # shape of gamma prior on latent event weights
beta = 0.01             # rate of gamma prior on latent event weights
concentration = 0.5     # concentration of Dirichlet prior on mark dist. for latent events

print("hyperparameters: ",\
    "min_event_length", MIN_EVENT_LENGTH,\
    "min_events_per_node", MIN_EVENTS_PER_NODE,\
    "max_events_per_node", MAX_EVENTS_PER_NODE,\
    "tau", tau, "mu", mu, "alpha", alpha, \
    "beta", beta, "concentration", concentration)

# Initialize the rate of background events based on expected number of induced events
lambda0 = max(N_mb - mu * alpha / beta, 1)

# bkgd_concentration = 1e-4
# bkgd_pis = np.zeros((M, D))
# for m in range(M):
#     bkgd_pis[m] = np.array(np.sum(ys[ms==m], axis=0))[0] + bkgd_concentration
#     bkgd_pis[m] /= bkgd_pis[m].sum()


bkgd_concentration = 1e-4
bkgd_pis = np.zeros((M, D))
for m in range(M):
    bkgd_pis[m] = np.array(np.sum(ys_all[ms_all==midx2entity[m]], axis=0))[0][valid_word_idxs] + bkgd_concentration
    bkgd_pis[m] /= bkgd_pis[m].sum()


# In[25]:

# calculate the empirical probability of each node
# uniq_node, node_freq = np.unique(ms_mb, return_counts=True)
# order = np.argsort(uniq_node)
# node_freq = node_freq[order]
# uniq_node = uniq_node[order]
# node_prob = np.zeros(M)
# node_prob[uniq_node] = node_freq
# node_prob = node_prob / node_prob.sum() + 1e-20

# load background rates from past em
bg_rates = np.load('../mfm/bg_rates_final_params.npy')
node_freq = np.array([bg_rates[bg_rates[:,0]==midx2entity[i],1][0] for i in range(M)])
node_prob = node_freq / node_freq.sum() + 1e-20


# excite_wts = np.load('../mfm/excite_wts_final_params.npy')

# Specify the background model
background = NodeAndTimeAndMarkBackground(
    num_nodes=M,
    node_distribution=node_prob,
    time_class=UniformTimeBackground,
    time_kwargs=dict(T=T),
    mark_class=MultinomialBackground,
    mark_kwargs=dict(data_dim=D),
    )

# Set the background rates
alpha = 1.
for m, bkgd_mark_dist in enumerate(background.mark_backgrounds):
    bkgd_mark_dist.concentration = bkgd_pis[m] * alpha

# Specify the observation model
obs_class = NodeAndTimeAndMarkCluster
obs_hypers = dict(
    num_nodes=M,
    node_concentration=0.5,
    time_class=ExponentialTimeCluster,
    time_kwargs=dict(T=T, tau=tau),
    mark_class=MultinomialCluster,
    mark_kwargs=dict(data_dim=D, concentration=concentration)
    )
model = NeymanScottModel(mu, alpha, beta, lambda0, background, obs_class, observation_hypers=obs_hypers)


# In[ ]:

print(data.shape)

# Fit the model with Gibbs
num_iters = 20
# samples = model.fit(data[npr.choice(np.arange(data.shape[0]), 1000, replace=False)], num_iters=num_iters, init_method="background")

# samples = model.fit(data, num_iters=num_iters, init_method="background")
samples = model.fit(data, method="mcem", step_size=0.1, num_iters=5, num_gibbs_samples=2, verbose=True)

# Extract the number of clusters for each sample
t_samples = np.array([s["num_clusters"] for s in samples])


# In[ ]:


# plt.figure(figsize=(8, 8))

# plt.subplot(313)
# plt.imshow([s["parents"] for s in samples], cmap="jet", aspect="auto")
# plt.title("Parent samples")
# plt.ylabel("Gibbs Iteration")
# plt.xlabel("Event Index (sorted by time)")
# plt.colorbar()

# # Plot the number of clusters over samples
# plt.figure()
# plt.plot([s["num_clusters"] for s in samples], label="sampled")
# plt.xlabel("Iteration")
# plt.ylabel("Num clusters")
# plt.legend(loc="lower right")
# plt.show()


# In[ ]:


samples[-1]["clusters"]


# In[ ]:


# Investigate the latent events
n_words = 25
clusters = samples[-1]["clusters"]
cluster_sizes = [c.size for c in clusters]
cluster_perm = np.argsort(cluster_sizes)[::-1]
for i,k in enumerate(cluster_perm):
    cluster = clusters[k]
    t_event = cluster.time_cluster.t_min
    date_event = start_date + timedelta(days=t_event)
    a_event = cluster.mark_cluster.a
    topN = np.argsort(a_event)[::-1][:n_words]
    print("Cluster {}. Size: {}. Time: {}. Top {} words:".format(k, cluster.size, date_event.strftime("%b %d, %Y"), n_words))
    print([yidx2term[idx] for idx in topN ])
    print("")


# In[ ]:


def investigate_cluster(cidx, parents):
    for event_idx in np.where(parents == cidx)[0]:
        date_event = start_date + timedelta(days=ts_mb[event_idx])
        print("Event {}. Time: {}".format(event_idx, date_event.strftime("%b %d, %Y")))
        print([yidx2term[v] for v in ys_mb[event_idx].nonzero()[1]])
        print("")

investigate_cluster(1, samples[-1]["parents"])

# import time
# np.save(os.path.join(outdir, str(int(time.time()))+"fit.npy"), samples[-1]["parents"])


# In[ ]:




