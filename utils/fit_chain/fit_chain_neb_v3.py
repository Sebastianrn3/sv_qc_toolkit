import numpy as np
from utils.io.run_1scf import main_mopac_1d
from settings.config import EPS

#CI neatsinaujina
#rigid + fixed - ivesti
dt = 0.2
trust_radius = 0.15
k_of_spring = 0.1

lbfgs_m = 10 #memory, 20
lbfgs_curv_eps = 1e-12

geff_tol = 1e-3
ci_start_step = 10  # 5-20

use_ew_springs = True
k_l = 0.05
k_u = 0.20

def fit_chain_neb_ci(images, atoms, n_steps, cfg):
    images = [np.asarray(x).copy() for x in images]
    n_img = len(images)

    st = {}
    reset_lbfgs(st)

    ci_index = None
    energies_last = None
    converged = False

    for step in range(n_steps):
        energies = np.zeros(n_img)
        grads = [None] * n_img

        #1 mopac
        for i, image in enumerate(images):
            E, grad = main_mopac_1d(atoms, image, cfg)
            energies[i], grads[i] = E, np.asarray(grad).reshape(-1)

        energies_last = energies.copy()

        ksp = compute_ew_ksp(energies) if use_ew_springs else np.full(n_img, k_of_spring)

        #2 switch CI once
        if ci_index is None and step >= ci_start_step:
            ci_index = 1 + np.argmax(energies[1:-1])

        print_barriers(energies, ksp, step)

        #3 NEB force
        geff_list = compute_geff_list(images, energies, grads, ci_index, ksp)
        max_geff = max(np.linalg.norm(geff_list[i]) for i in range(1, n_img - 1))
        print(f"step ------------------------------------"
              f"{step:4d}\n | max||g_eff||={max_geff:.3e} | CI={ci_index}")

        if max_geff < geff_tol:
            converged = True
            break

        #4 GLBFGS step
        X = pack_internal(images)
        G = pack_geff(geff_list)

        update_lbfgs(st, X, G, m=lbfgs_m, curv_eps=lbfgs_curv_eps)
        P = direct_lbfgs(st, G)

        dx = dt * P
        nrm = np.linalg.norm(dx)
        if nrm > trust_radius:
            dx *= trust_radius / (nrm + EPS)

        X_new = X + dx
        internal_new = unpack_internal(X_new, images)

        images = [images[0].copy()] + internal_new + [images[-1].copy()] #rebuild

    return images, energies_last

# ---------- NEB geometry helpers

def get_unit_vector(v):
    v = np.asarray(v).reshape(-1)
    return v / (np.linalg.norm(v) + EPS)

def get_tangent(R, energies, i):
    E_prev, Ei, E_next = energies[i - 1], energies[i], energies[i + 1]
    d_prev, d_next = R[i] - R[i - 1], R[i + 1] - R[i]

    if (E_next > Ei) and (Ei >= E_prev):
        t = d_next
    elif (E_prev > Ei) and (Ei >= E_next):
        t = d_prev
    else:
        dE_plus, dE_minus = abs(E_next - Ei), abs(E_prev - Ei)
        t = dE_plus * d_next + dE_minus * d_prev if (E_next > E_prev) else dE_minus * d_next + dE_plus * d_prev

    return get_unit_vector(t)

def compute_geff_list(images, energies, grads, ci_index, ksp):
    R = [np.asarray(x).reshape(-1) for x in images]
    n_img = len(R)
    geff = [None] * n_img

    for i in range(1, n_img - 1):
        Fi = -np.asarray(grads[i]).reshape(-1)
        tau = get_tangent(R, energies, i)

        if (ci_index is not None) and (i == ci_index):
            F_eff = Fi - 2 * np.dot(Fi, tau) * tau
        else:
            d_forward = np.linalg.norm(R[i + 1] - R[i])
            d_backward = np.linalg.norm(R[i] - R[i - 1])
            F_perp = Fi - np.dot(Fi, tau) * tau

            ki = float(ksp[i])
            F_spring = ki * (d_forward - d_backward) * tau

            F_eff = F_perp + F_spring

        geff[i] = -F_eff

    return geff

# ---------- packing
def pack_internal(images):
    return np.concatenate([np.asarray(img).reshape(-1) for img in images[1:-1]])

def unpack_internal(x, template_images):
    internal = []
    off = 0
    for image in template_images[1:-1]:
        n = np.asarray(image).size
        internal.append(x[off:off + n].reshape(-1, 3))
        off += n
    return internal

def pack_geff(geff_list):
    return np.concatenate([np.asarray(g).reshape(-1) for g in geff_list[1:-1]])

# ---------- L-BFGS
def reset_lbfgs(st):
    st["S"], st["Y"], st["RHO"] = [], [], []
    st["x_prev"], st["g_prev"] = None, None

def update_lbfgs(st, x, g, m=10, curv_eps=1e-12):
    x = np.asarray(x).reshape(-1)
    g = np.asarray(g).reshape(-1)

    if st["x_prev"] is None:
        st["x_prev"] = x.copy()
        st["g_prev"] = g.copy()
        return

    s = x - st["x_prev"]
    y = g - st["g_prev"]
    ys = np.dot(y, s)

    st["x_prev"] = x.copy()
    st["g_prev"] = g.copy()

    if ys <= curv_eps:
        return

    st["S"].append(s)
    st["Y"].append(y)
    st["RHO"].append(1 / (ys + EPS))

    if len(st["S"]) > m:
        st["S"].pop(0); st["Y"].pop(0); st["RHO"].pop(0)

def direct_lbfgs(st, g):
    g = np.asarray(g).reshape(-1)
    S, Y, RHO = st["S"], st["Y"], st["RHO"]

    q = g.copy()
    alpha = [0.0] * len(S)

    for i in range(len(S) - 1, -1, -1): #1st 2loop recursion
        alpha[i] = RHO[i] * np.dot(S[i], q) #a=rho(sq)
        q -= alpha[i] * Y[i]

    if len(S) > 0:
        s_last, y_last = S[-1], Y[-1]
        gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + EPS)
    else:
        gamma = 1

    r = gamma * q #r=H0q≈γIq

    for i in range(len(S)):#2nd 2loop recursion
        beta = RHO[i] * np.dot(Y[i], r) #beta=rho(y⋅r)
        r += S[i] * (alpha[i] - beta) #r←r+s(alpha-beta)

    return -r #≈-H(k)g(k)
#---------------
def compute_ew_ksp(energies, k_l=k_l, k_u=k_u):
    E = np.asarray(energies)
    n = len(E)

    Eref = max(E[0], E[-1])
    Emax = np.max(E)
    k_spring = np.full(n, k_l)

    for i in range(1, n - 1):
        if E[i] > Eref:
            alpha = (Emax - E[i])/(Emax - Eref)
            k_spring[i] = (1.0 - alpha) * k_u + alpha * k_l
        else:
            k_spring[i] = k_l

    return k_spring

#misc
def print_barriers(energies, ksp, step):
    Eint_max = float(np.max(energies[1:-1]))
    Ea_plus = Eint_max - float(energies[0])
    Ea_minus = Eint_max - float(energies[-1])
    print(f"Ea+ = {Ea_plus:.8f}")
    print(f"Ea- = {Ea_minus:.8f}")

    if use_ew_springs and step % 5 == 0:
        i_max = 1 + int(np.argmax(energies[1:-1]))
        print(f"EW: Eref={max(energies[0], energies[-1]):.6f} "
              f"Emax={np.max(energies[1:-1]):.6f} "
              f"k(CI-cand @ i={i_max})={ksp[i_max]:.4f}")