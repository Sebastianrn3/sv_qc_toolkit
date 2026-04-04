import time
import numpy as np
from itertools import product

def evaluate_triplet_score(x_prev, x_i, x_next, g_i, EPS=1e-4):
    s = (x_prev - x_next).reshape(-1, 3)
    g = g_i.reshape(-1, 3)

    ns = np.sqrt(np.sum(s * s, axis=1))
    ng = np.sqrt(np.sum(g * g, axis=1))

    mask = (ns > EPS) & (ng > EPS)
    if not np.any(mask):
        cos_value = 0.0
    else:
        dot = np.abs(np.sum(s[mask] * g[mask], axis=1))
        cos_value = np.mean(dot / (ns[mask] * ng[mask]))

    log = {
        "cos": float(cos_value),
    }
    return float(cos_value), log


def even_distribution_score(chain_geoms):
    X = [np.asarray(x).reshape(-1, 3) if np.asarray(x).ndim == 1 else np.asarray(x)
         for x in chain_geoms]

    n = len(X)
    if n < 3:
        return 0.0

    seg = np.array([
        np.linalg.norm((X[i] - X[i-1]).ravel())
        for i in range(1, n)
    ], dtype=float)

    total = seg.sum()
    if total < 1e-12:
        return 0.0

    cum = np.concatenate([[0.0], np.cumsum(seg)]) / total
    m = n - 1
    tol = 0.5 / m

    score = 0.0
    for i in range(1, n - 1):
        target = i / m
        score += max(0.0, 1.0 - abs(cum[i] - target) / tol)

    return float(score)

def brute_force_paths(groups):
    #-----------supportive functions start
    # -----------supportive functions end
    N = len(groups)-2 #interpoliants only
    K = len(groups[1])
    print("Images in chain:",N,"+2", " Selected:", K, " Combos:", K**N)

    X = [[np.asarray(it["geom"]).ravel() for it in grp] for grp in groups]
    G = [[np.asarray(it["grad"]).ravel() for it in grp] for grp in groups]

    r_index, p_index = len(groups[0]) - 1, len(groups[-1])- 1

    variants=list(product([i for i in range(K)],repeat=N))
    scores=np.zeros(len(variants))
    t0 = time.time()
    top_scores = []
    cos_all = []
    cos2_all = []
    counter_it = 0
    for iteration, combo in enumerate(variants):
        counter_it += 1
        if counter_it % 10000 == 0:
            print(counter_it, time.time()-t0)
        acc=0.0
        recombo=[r_index]+list(combo)+[p_index]
        for t in range(1, N + 1):
            x_prev=X[t-1][recombo[t-1]]
            x_i=X[t][recombo[t]]
            x_next=X[t+1][recombo[t+1]]
            g_i = G[t][recombo[t]]

            score_value, log = evaluate_triplet_score(x_prev, x_i, x_next, g_i)

            acc += score_value

            cos_all.append(log["cos"])

        chain_geoms = [X[t][recombo[t]] for t in range(len(groups))]
        even_score = even_distribution_score(chain_geoms)
        acc += even_score

        scores[iteration] = acc
        full_combo = (r_index, *combo, p_index)
        top_scores.append((acc, combo, full_combo))

#===resuming
    top_scores.sort(key=lambda x: x[0], reverse=True)
    topn = min(5, len(top_scores))
    top5 = top_scores[:topn]

    print(f"Meaningfuls are {len(top_scores)}/{K ** N}")

    print("Top combos:")
    for rank, (score, combo, full_combo) in enumerate(top5, start=1):
        print(
            f"{rank}. score = {score:.6f}, "
            f"inner combo = {combo}, "
            f"full combo = {full_combo}"
        )

    print("\nDiagnostics:")
    if len(cos_all) > 0:
        cos_arr = np.asarray(cos_all, dtype=float)
        abs_cos = np.abs(cos_arr)

        print(f"cos median               = {np.median(cos_arr):.4g}")
        print(f"abs(cos) median          = {np.median(abs_cos):.4g}")
        print(f"abs(cos) p95             = {np.percentile(abs_cos, 95):.3e}")
        print(f"abs(cos) mean            = {np.mean(abs_cos):.4g}")
        print(f"abs(cos) min/max         = {np.min(abs_cos):.4g} / {np.max(abs_cos):.4g}")
    else:
        print("No cosine diagnostics collected.")

    print("\nTop indices in raw score array:")
    top_idx = np.argsort(scores)[-topn:][::-1]
    for rank, idx in enumerate(top_idx, start=1):
        print(
            f"{rank}. idx = {idx}, "
            f"score = {scores[idx]:.6f}, "
            f"combo = {variants[idx]}"
        )

    return top_scores[0] if top_scores else None


def analyze_chain_triplets(groups, full_combo, out=None):
    def p(*args, **kwargs):
        if out is None:
            print(*args, **kwargs)
        else:
            print(*args, file=out, **kwargs)

    def angle_deg(c):
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    p("full combo:", full_combo)

    n_groups = len(groups)
    total_score = 0

    for t in range(1, n_groups - 1):
        item_prev = groups[t - 1][full_combo[t - 1]]
        item_i    = groups[t][full_combo[t]]
        item_next = groups[t + 1][full_combo[t + 1]]

        x_prev = np.asarray(item_prev["geom"])
        x_i    = np.asarray(item_i["geom"])
        x_next = np.asarray(item_next["geom"])
        g_i    = np.asarray(item_i["grad"])
        g_f      = g_i.ravel()

        d_prev   = x_i.ravel() - x_prev.ravel()
        d_next   = x_next.ravel() - x_i.ravel()
        tangent  = x_prev.ravel() - x_next.ravel()

        norm_g       = np.linalg.norm(g_f)
        norm_tangent = np.linalg.norm(tangent)
        norm_d_prev  = np.linalg.norm(d_prev)
        norm_d_next  = np.linalg.norm(d_next)

        if norm_g * norm_tangent < 1e-10:
            cos_value = 1
            score_value = 1
        else:
            cos_value = (g_f @ tangent) / (norm_g * norm_tangent)
            score_value = cos_value ** 2

        total_score += score_value

        E_prev = item_prev["energy"]
        E_i    = item_i["energy"]
        E_next = item_next["energy"]

        if norm_tangent < 1e-15:
            g_parallel_norm = np.nan
            g_perp_norm = np.nan

        else:
            tau = tangent / norm_tangent
            g_parallel = (g_f @ tau) * tau
            g_perp = g_f - g_parallel
            g_parallel_norm = np.linalg.norm(g_parallel)
            g_perp_norm = np.linalg.norm(g_perp)


        g_atoms = g_i.reshape(-1, 3)
        t_atoms = (x_prev - x_next).reshape(-1, 3)

        atom_dot = np.sum(g_atoms * t_atoms, axis=1)
        atom_abs_dot = np.abs(atom_dot)
        atom_t_norm = np.linalg.norm(t_atoms, axis=1)
        atom_g_norm = np.linalg.norm(g_atoms, axis=1)

        atom_cos = np.full(len(atom_dot), np.nan)
        atom_mask = (atom_g_norm > 1e-15) & (atom_t_norm > 1e-15)
        atom_cos[atom_mask] = atom_dot[atom_mask] / (atom_g_norm[atom_mask] * atom_t_norm[atom_mask])

        p("---------")
        p(f"bead {t}/{len(full_combo)-2}, k={full_combo[t]}, angle {angle_deg(cos_value):.3g} deg")
        p(f"  triplet_score, cos  = {score_value:.4g}, {cos_value:.4g}")
        p(f" E_i, dE(i-1->i->i+1) = {E_i:3g}, {(E_i - E_prev):.2f} -> {(E_next - E_i):.2f}")
        p(f"norms: |g|,|prev->next|,|prev->next|      = {norm_g:.5g}, {norm_d_prev:.5g}, {norm_d_next:.5g}")
        p(f"g decomp-to-tangent: |g_parallel|,|g_perp|= {g_parallel_norm:.5g}, {g_perp_norm:.5g}")
        p(f"atom's sum(|g⋅t|) / sum(g⋅t)               = { np.abs(np.sum(atom_dot)) / np.sum(atom_abs_dot):.5g}")
    p("======")
    p(f"Total chain score = {total_score:.5g}")