"""Message passing functions for site percolation, shared by the new notebooks.

The first functions were written and checked in notebook 03 (03_hybrid_phi_method.ipynb) and copied here
unchanged; the later ones (small f, compiled solvers, fit of the threshold) were added with notebooks 04-08.
Use: `from mp import message_passing, matrice_G, lambda_max, ...` from a notebook in the same folder.
Notebooks 01 and 03 keep their own cells and do not import this file.
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.optimize import curve_fit


# --- message passing and Monte Carlo (notebook cells 4-7) ---

def message_passing(G, phi, tol=1e-10, max_iter=10000, aggiornamento="inplace", t0=None):
    """Site percolation with message passing (course book v4, sec. 8.2.4, eqs. 8.24-8.26).

    phi: a number (the same for all nodes) or a dict {node: phi of that node}.
    aggiornamento: "inplace" (online, Gauss-Seidel, the default: each new message is used at once
    in the same sweep) or "sincrono" (offline, t_{n+1} = F(t_n)). Same fixed point; in-place needs
    about half the iterations (test_mp.ipynb).
    t0: starting messages {(i, j): t_{i->j}} (warm start, e.g. the fixed point of the intact network);
    if None, all messages start from 1.
    Returns S_MP, s (dict node -> s_i), t (dict (i, j) -> t_{i->j}),
    the number of iterations used and whether the iteration has converged.
    """
    #if nx.number_of_selfloops(G) > 0:
    #    raise ValueError("the network has self-loops: against the assumptions of the model")

    nodi = list(G.nodes())
    phi_n = {n: phi for n in nodi} if np.isscalar(phi) else dict(phi)

    # 2|E| messages: two directions for each link
    messaggi = [(i, j) for i, j in G.edges()] + [(j, i) for i, j in G.edges()]  # the "+" joins the two lists: first the links in the direction given by G.edges(), then reversed

    # for each message i->j, the list of messages it has to read
    # (j->k with k in N_j \ {i}) is computed only once, outside the loop.
    # It is the row i->j of the matrix G from the notes of 8 October.
    # In practice, here we build the 0s and 1s of the matrix G
    da_leggere = {(i, j): [(j, k) for k in G.neighbors(j) if k != i] for (i, j) in messaggi}

    if t0 is None:
        t = {m: 1.0 for m in messaggi}   # start far from t = 0, which is always a solution: starting from 0 we would stay there every time, starting from 1 we do not get stuck in it
    else:
        t = dict(t0)                     # warm start: a copy, so the dict passed in is not changed
    converge = False

    for it in range(1, max_iter + 1):
        if aggiornamento == "sincrono":
            # synchronous update: all the new messages from the old ones, t_{n+1} = F(t_n)
            t_new = {}
            for (i, j) in messaggi:
                prod = 1.0                   # empty product = 1: for a leaf j we get t_{i->j} = 0
                for m in da_leggere[(i, j)]:
                    prod *= 1.0 - t[m]      # this is the product: no link leads to the GCC
                t_new[(i, j)] = phi_n[j] * (1.0 - prod)          # eq. 8.25: at least one leads to the GCC

            # we stop when no message changes by more than tol
            diff = max((abs(t_new[m] - t[m]) for m in messaggi), default=0.0)
            t = t_new

        elif aggiornamento == "inplace":
            # in-place update: each new message is written in t at once,
            # so the messages after it in the same sweep already read the new value
            diff = 0.0
            for (i, j) in messaggi:
                prod = 1.0
                for m in da_leggere[(i, j)]:
                    prod *= 1.0 - t[m]
                nuovo = phi_n[j] * (1.0 - prod)                  # eq. 8.25, as above
                diff = max(diff, abs(nuovo - t[(i, j)]))         # change of this message, before overwriting it
                t[(i, j)] = nuovo

        else:
            raise ValueError(f"unknown update: {aggiornamento}")

        if diff < tol:
            converge = True
            break

    # eq. 8.24: one s_i per node (an isolated node has an empty product, so s_i = 0)
    s = {}
    for i in nodi:
        prod = 1.0
        for j in G.neighbors(i):
            prod *= 1.0 - t[(i, j)]
        s[i] = phi_n[i] * (1.0 - prod)

    S_MP = sum(s.values()) / len(nodi)   # eq. 8.26, divided by all the N nodes as in mc_1run
    return S_MP, s, t, it, converge


def mc_1run(G, phi, rng):
    """One run of site percolation. Returns S^MC of that run.

    phi: a number (the same for all nodes) or a dict {node: phi of that node},
    as in message_passing.
    """
    nodi = list(G.nodes)
    N = len(nodi)

    # one phi per node, in the same order as nodi
    phi_n = np.full(N, phi) if np.isscalar(phi) else np.array([phi[n] for n in nodi])

    # one value per node: True (1) if the node stays, with its own probability phi
    presente = rng.random(N) < phi_n
    rimasti = [n for n, c in zip(nodi, presente) if c]

    # limit case: no node left
    if len(rimasti) == 0:
        return 0.0

    # graph with only the nodes left and the links between them
    H = G.subgraph(rimasti)

    # connected groups, sorted from the largest to the smallest
    gruppi = sorted(nx.connected_components(H), key=len, reverse=True)
    lcc = gruppi[0]

    return len(lcc) / N


def curve_mc(G, num_runs, num_phis, rng):
    """Monte Carlo curve on a graph: num_phis values of phi between 0 and 1, num_runs runs for each.
    Returns the values of phi and a matrix with one row per phi and one column per run."""
    phis = np.linspace(0, 1, num_phis)

    results = []

    for i in range(len(phis)):
        intermediate = []
        for k in range(num_runs):
            intermediate.append(mc_1run(G, phis[i], rng))
        results.append(intermediate)

    return phis, np.array(results)


def scarto_quadratico_medio(S_a, S_b):
    """Square root of the mean of the squared differences between two curves computed on the same phi."""
    differenze = np.asarray(S_a) - np.asarray(S_b)
    return np.sqrt(np.mean(differenze ** 2))


# --- linearised message passing: matrix G and its largest eigenvalue (cell 8) ---

def matrice_G(G):
    """Matrix G of the linearised message passing (notes, Step 5; course book v4, p. 246).
    Rows and columns are the 2|E| messages; G[i->j, j->k] = 1 for every k in N_j other than i.
    Returns G as a sparse CSR matrix and the list of the messages, in the order of the rows."""
    messaggi = [(i, j) for i, j in G.edges()] + [(j, i) for i, j in G.edges()]   # same order as in message_passing
    indice = {m: n for n, m in enumerate(messaggi)}     # message -> its row (and column) number

    righe, colonne = [], []
    for (i, j) in messaggi:
        for k in G.neighbors(j):          # the same loop as da_leggere
            if k != i:
                righe.append(indice[(i, j)])
                colonne.append(indice[(j, k)])

    M = len(messaggi)
    valori = np.ones(len(righe))          # all ones, as floats: eigs works with floating point numbers
    G_mat = sp.csr_matrix((valori, (righe, colonne)), shape=(M, M))
    return G_mat, messaggi


def lambda_max(G_mat):
    """Largest eigenvalue of the sparse matrix G (scipy.sparse.linalg.eigs, ARPACK)."""
    val = eigs(G_mat, k=1, which="LR", return_eigenvectors=False)
    return val[0].real


# --- hybrid phi and screening (cells 18, 22, 30, 36, 41, 51, 71) ---

def estrai_protetti(G, frazione, protocollo, rng, betweenness=None):
    """Nodes to screen: the fraction `frazione` of all the nodes, chosen by `protocollo`
    ("random", "degree" or "betweenness"). Ties are broken at random.
    betweenness: dict {node: betweenness} computed once per network with nx.betweenness_centrality;
    if None it is computed here (slow on large networks).
    Returns the list of the screened nodes."""
    num_protetti = int(round(frazione * G.number_of_nodes()))

    # random order of the nodes: for "random" we take the first ones,
    # for the other protocols it decides the order among nodes with the same score
    nodi = list(G.nodes)
    rng.shuffle(nodi)

    if protocollo == "random":
        return nodi[:num_protetti]
    elif protocollo == "degree":
        punteggio = dict(G.degree)
    elif protocollo == "betweenness":
        punteggio = betweenness if betweenness is not None else nx.betweenness_centrality(G)
    else:
        raise ValueError(f"unknown protocol: {protocollo}")

    # sorted keeps the order of the list among equal scores, so the ties stay random
    ordinati = sorted(nodi, key=lambda n: punteggio[n], reverse=True)
    return ordinati[:num_protetti]


def phi_ibrido(G, protetti, phi):
    """Hybrid phi: 1 for the screened nodes, phi for all the others.
    Returns the dict {node: phi of that node}, as accepted by message_passing and mc_1run."""
    phi_n = {n: phi for n in G.nodes}
    for n in protetti:
        phi_n[n] = 1.0
    return phi_n


def phi_efficace(phi, frazione):
    """phi_eff = phi + f (1 - phi): probability that a node is present when the fraction f of the nodes,
    chosen at random, is screened (Artime and De Domenico, arXiv:2104.14893, SI.1)."""
    return phi + frazione * (1 - phi)


_destinazioni = {}      # id(messaggi) -> (messaggi, nodes, index of the node each message points to)


def matrice_M(G_mat, messaggi, phi_n):
    """M = diag(phi) G of the linearised message passing with node-dependent phi.
    G_mat, messaggi: from matrice_G, built once per network; phi_n: dict {node: phi of that node}.
    The row of the message i->j is multiplied by phi_j. Returns M as a sparse matrix.
    The index "node j of each message i->j" is built once per list of messages and kept in _destinazioni,
    so at each call only the N values of phi are read from the dict (not the 2|E| messages)."""
    voce = _destinazioni.get(id(messaggi))
    if voce is None or voce[0] is not messaggi:
        nodi = list(dict.fromkeys(j for (i, j) in messaggi))
        pos = {n: a for a, n in enumerate(nodi)}
        voce = (messaggi, nodi, np.array([pos[j] for (i, j) in messaggi], dtype=np.int64))
        _destinazioni[id(messaggi)] = voce
    _, nodi, dest = voce
    phi_nodi = np.fromiter((phi_n[n] for n in nodi), dtype=float, count=len(nodi))
    return sp.diags(phi_nodi[dest]) @ G_mat      # one value per message, same order as the rows


def lambda_zero(g, protetti):
    """lambda(0): largest eigenvalue of G built on the subgraph of the screened nodes alone."""
    H = g.subgraph(protetti)
    if H.number_of_edges() == 0:
        return 0.0
    G_H = matrice_G(H)[0]
    if G_H.shape[0] < 2000:
        return np.max(np.linalg.eigvals(G_H.toarray()).real)
    return lambda_max(G_H)


def soglia_bisezione(g, G_mat, messaggi, protetti, tol=1e-4):
    """Critical threshold phi_c with hybrid phi: lambda(phi) = 1, with lambda the largest eigenvalue of
    M = diag(phi) G, the screened nodes at phi = 1 and all the others at phi.
    G_mat, messaggi: from matrice_G, built once per network. Returns phi_c (0 if the screened nodes alone
    percolate, nan if the network does not percolate even when intact)."""
    def lam(phi):
        M = matrice_M(G_mat, messaggi, phi_ibrido(g, protetti, phi))
        if M.count_nonzero() == 0:      # all entries zero (e.g. phi = 0 and no screened node): eigs fails, lambda = 0
            return 0.0
        return lambda_max(M)

    # special cases
    if lambda_zero(g, protetti) >= 1:
        return 0.0
    if lam(1.0) < 1:
        return np.nan

    basso, alto = 0.0, 1.0
    while alto - basso > tol:
        medio = (basso + alto) / 2
        if lam(medio) < 1:          # below the threshold: the crossing is in the upper half
            basso = medio
        else:                       # above the threshold: the crossing is in the lower half
            alto = medio
    return (basso + alto) / 2


def budget_critico(g, ordine):
    """Smallest number of nodes, taken from the start of the list `ordine`, that percolate alone at phi = 0
    (lambda(0) >= 1). Returns nan if even all the nodes do not percolate."""
    if lambda_zero(g, ordine) < 1:
        return np.nan
    basso, alto = 0, len(ordine)        # lambda(0) < 1 with the first `basso` nodes, >= 1 with the first `alto`
    while alto - basso > 1:
        medio = (basso + alto) // 2
        if lambda_zero(g, ordine[:medio]) >= 1:
            alto = medio
        else:
            basso = medio
    return alto


def lcc_protetti(g, protetti):
    """Largest component of the subgraph induced by the screened nodes alone, as a fraction of N."""
    if len(protetti) == 0:
        return 0.0
    return len(max(nx.connected_components(g.subgraph(protetti)), key=len)) / g.number_of_nodes()


# --- small f, draws and non-uniform phi grid (29 Oct 2026) ---

def top_unico(G, frazione, protocollo, betweenness=None):
    """True if the screened nodes of the protocol are always the same set, whatever the order of the ties:
    the score of the last screened node is strictly higher than the score of the first node left out
    (the order inside the set does not matter). Always False for "random" with 0 < n < N."""
    N = G.number_of_nodes()
    n = int(round(frazione * N))
    if n == 0 or n >= N:
        return True
    if protocollo == "random":
        return False
    punteggio = dict(G.degree) if protocollo == "degree" else betweenness
    valori = sorted(punteggio.values(), reverse=True)
    return valori[n - 1] > valori[n]


def num_estrazioni(G, frazione, protocollo, betweenness=None, base=30, minimo=10):
    """Number of draws of the screened nodes: 1 if the set is always the same (top_unico),
    otherwise round(base / n) + minimo, with n the number of screened nodes."""
    if top_unico(G, frazione, protocollo, betweenness):
        return 1
    n = int(round(frazione * G.number_of_nodes()))
    return int(round(base / n)) + minimo


def griglia_phi(phi_c, larghezza=0.1, passo_fine=0.01, passo_largo=0.05):
    """Grid of phi in [0, 1]: step passo_fine in [phi_c - larghezza, phi_c + larghezza], passo_largo elsewhere."""
    largo = np.arange(0, 1 + 1e-9, passo_largo)
    fine = np.arange(max(0.0, phi_c - larghezza), min(1.0, phi_c + larghezza) + 1e-9, passo_fine)
    return np.unique(np.round(np.concatenate([largo, fine]), 4))


def scarto_quadratico_integrale(phis, S_a, S_b):
    """RMS deviation between two curves on a grid that may be non-uniform: sqrt of the integral of the
    squared difference (trapezoidal rule), divided by the length of the interval.
    On a uniform grid it is close to scarto_quadratico_medio."""
    phis = np.asarray(phis)
    differenze = np.asarray(S_a) - np.asarray(S_b)
    return np.sqrt(np.trapezoid(differenze ** 2, phis) / (phis[-1] - phis[0]))


# --- message passing compiled with numba (29 Oct 2026) ---
# Same equations and same in-place update as message_passing, on arrays instead of dicts.
# The messages are numbered as in message_passing and matrice_G (G.edges(), then reversed);
# for the message m = i->j, the messages it reads (j->k, k in N_j \ {i}) are
# leggi[inizio[m]:inizio[m + 1]], the same structure as the rows of G in CSR form.

from numba import njit


def struttura_mp(G):
    """Arrays for the compiled solver, built once per network: list of the messages, index of the node each
    message points to (j for i->j), CSR structure of the messages to read, CSR structure of the messages
    leaving each node (i->j for j in N_i, for s_i), and the list of the nodes."""
    nodi = list(G.nodes())
    pos = {n: a for a, n in enumerate(nodi)}
    messaggi = [(i, j) for i, j in G.edges()] + [(j, i) for i, j in G.edges()]
    indice = {m: a for a, m in enumerate(messaggi)}

    destinazione = np.array([pos[j] for (i, j) in messaggi], dtype=np.int64)
    inizio, leggi = [0], []
    for (i, j) in messaggi:
        leggi.extend(indice[(j, k)] for k in G.neighbors(j) if k != i)
        inizio.append(len(leggi))
    inizio_s, uscenti = [0], []
    for i in nodi:
        uscenti.extend(indice[(i, j)] for j in G.neighbors(i))
        inizio_s.append(len(uscenti))
    return (messaggi, nodi, destinazione, np.array(inizio, dtype=np.int64), np.array(leggi, dtype=np.int64),
            np.array(inizio_s, dtype=np.int64), np.array(uscenti, dtype=np.int64))


@njit(cache=True)
def _itera_inplace(t, phi_dest, inizio, leggi, tol, max_iter):
    """In-place iterations of eq. 8.25 on the array t (changed in place). Returns (iterations, converged)."""
    for it in range(1, max_iter + 1):
        diff = 0.0
        for m in range(t.shape[0]):
            prod = 1.0
            for a in range(inizio[m], inizio[m + 1]):
                prod *= 1.0 - t[leggi[a]]
            nuovo = phi_dest[m] * (1.0 - prod)
            d = abs(nuovo - t[m])
            if d > diff:
                diff = d
            t[m] = nuovo
        if diff < tol:
            return it, True
    return max_iter, False


@njit(cache=True)
def _calcola_s(t, phi_nodi, inizio_s, uscenti):
    """eq. 8.24: s_i = phi_i [1 - prod over the messages leaving i of (1 - t)]."""
    s = np.empty(phi_nodi.shape[0])
    for i in range(phi_nodi.shape[0]):
        prod = 1.0
        for a in range(inizio_s[i], inizio_s[i + 1]):
            prod *= 1.0 - t[uscenti[a]]
        s[i] = phi_nodi[i] * (1.0 - prod)
    return s


def message_passing_veloce(G, phi, tol=1e-10, max_iter=10000, t0=None, struttura=None):
    """Same as message_passing with aggiornamento="inplace", compiled with numba.
    struttura: from struttura_mp(G), to build it only once per network (built here if None).
    t0: starting messages, a dict {(i, j): t} as in message_passing or an array in the order of the messages.
    Returns S_MP, s (array, in the order of the nodes), t (array, in the order of the messages),
    the number of iterations used and whether the iteration has converged."""
    if struttura is None:
        struttura = struttura_mp(G)
    messaggi, nodi, destinazione, inizio, leggi, inizio_s, uscenti = struttura
    phi_nodi = np.full(len(nodi), float(phi)) if np.isscalar(phi) else np.array([phi[n] for n in nodi], dtype=float)
    phi_dest = phi_nodi[destinazione]

    if t0 is None:
        t = np.ones(len(messaggi))
    elif isinstance(t0, dict):
        t = np.array([t0[m] for m in messaggi], dtype=float)
    else:
        t = np.array(t0, dtype=float)          # a copy: the array passed in is not changed

    it, converge = _itera_inplace(t, phi_dest, inizio, leggi, tol, max_iter)
    s = _calcola_s(t, phi_nodi, inizio_s, uscenti)
    return s.sum() / len(nodi), s, t, it, converge


# --- Monte Carlo compiled with numba (29 Oct 2026) ---
# Same run as mc_1run (same random numbers, same present nodes, same S); only the largest component is found
# with a union-find on the edge list instead of networkx subgraphs.

def struttura_mc(G):
    """Arrays for mc_1run_veloce, built once per network: the list of the nodes and the two ends of each link,
    as positions in that list."""
    nodi = list(G.nodes)
    pos = {n: a for a, n in enumerate(nodi)}
    u = np.array([pos[a] for a, b in G.edges()], dtype=np.int64)
    v = np.array([pos[b] for a, b in G.edges()], dtype=np.int64)
    return nodi, u, v


@njit(cache=True)
def _lcc_unionfind(presente, u, v):
    """Size of the largest connected component of the present nodes. Union-find: every node starts as its own
    group; each link between two present nodes joins their two groups (the smaller one under the larger one);
    padre[x] points towards the root of the group of x, dim[root] is the size of the group."""
    N = presente.shape[0]
    padre = np.arange(N)
    dim = np.ones(N, np.int64)
    for a in range(u.shape[0]):
        x, y = u[a], v[a]
        if not (presente[x] and presente[y]):
            continue
        while padre[x] != x:              # root of x (path halving: each node jumps to its grandparent)
            padre[x] = padre[padre[x]]
            x = padre[x]
        while padre[y] != y:
            padre[y] = padre[padre[y]]
            y = padre[y]
        if x != y:                        # different groups: join them
            if dim[x] < dim[y]:
                x, y = y, x
            padre[y] = x
            dim[x] += dim[y]
    piu_grande = 0
    for i in range(N):
        if presente[i] and padre[i] == i and dim[i] > piu_grande:
            piu_grande = dim[i]
    return piu_grande


def mc_1run_veloce(G, phi, rng, struttura=None):
    """Same as mc_1run (same random numbers, same result), with the largest component found by union-find.
    phi: a number or a dict {node: phi of that node}; struttura: from struttura_mc(G) (built here if None)."""
    if struttura is None:
        struttura = struttura_mc(G)
    nodi, u, v = struttura
    N = len(nodi)
    phi_n = np.full(N, phi) if np.isscalar(phi) else np.array([phi[n] for n in nodi])
    presente = rng.random(N) < phi_n
    return _lcc_unionfind(presente, u, v) / N


def curva_mp(G, valori_phi, protetti=(), struttura=None):
    """S_MP on a grid of phi (hybrid phi: the nodes in `protetti` at 1, all the others at each value of
    valori_phi), with message_passing_veloce. The values are solved from the highest down, each solve
    starting from the messages of the previous, higher phi (warm start: they lie above the new fixed point,
    so the iteration comes down to it and does not stop at t = 0 above the threshold).
    Returns the array of S_MP in the order of valori_phi."""
    if struttura is None:
        struttura = struttura_mp(G)
    valori_phi = np.asarray(valori_phi, dtype=float)
    S = np.empty(len(valori_phi))
    t_prec = None
    for k in np.argsort(-valori_phi, kind="stable"):
        phi_n = phi_ibrido(G, protetti, valori_phi[k]) if len(protetti) else valori_phi[k]
        S[k], _, t_prec, _, _ = message_passing_veloce(G, phi_n, t0=t_prec, struttura=struttura)
    return S


def power_law(phi, A, phi_fit, beta):
    """S = A (phi - phi_fit)^beta, the curve of fit_soglia."""
    return A * (phi - phi_fit) ** beta


def fit_soglia(phi, S, S_min, S_max, min_punti=5):
    """Fit of S = A (phi - phi_fit)^beta on the points with S_min <= S <= S_max (notebook 07, section 1).
    Returns a dictionary with A, phi_fit, beta, the RMS of the residuals, the number of points and a note."""
    phi, S = np.asarray(phi, dtype=float), np.asarray(S, dtype=float)
    dentro = (S >= S_min) & (S <= S_max)
    risultato = {"A": np.nan, "phi_fit": np.nan, "beta": np.nan, "rms": np.nan, "points": int(dentro.sum()), "note": ""}
    if dentro.sum() < min_punti:
        risultato["note"] = "too few points"
        return risultato
    x, y = phi[dentro], S[dentro]
    try:
        p, _ = curve_fit(power_law, x, y, p0=[1.0, x.min() / 2, 1.1],
                         bounds=([0.0, -0.5, 0.2], [5.0, x.min() - 1e-3, 4.0]))
    except RuntimeError:
        risultato["note"] = "no convergence"
        return risultato
    risultato.update(A=p[0], phi_fit=p[1], beta=p[2], rms=np.sqrt(np.mean((power_law(x, *p) - y) ** 2)))
    return risultato
