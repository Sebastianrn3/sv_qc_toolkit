import numpy as np
from utils.io.run_1scf import main_mopac_1d

EPS = 1e-12

def fit_chain_simplified_string(images, atoms, n_steps,
    dt = 0.05,
    trust_radius: float = 0.15,
    use_preconditioner: bool = True,
    damping: float = 1e-3
):
    images = [np.asarray(x).copy() for x in images]
    preconditioners = None

    for step in range(n_steps):
        images, energies, grads, preconditioners = simplified_string_step(
            images,
            atoms,
            dt=dt,
            trust_radius=trust_radius,
            use_preconditioner=use_preconditioner,
            damping=damping,
            preconditioners=preconditioners,
        )

        maxg = max(np.linalg.norm(np.asarray(g).reshape(-1)) for g in grads)
        score = sum_chain_score(images, grads)
        print(
            f"\nstep {step}"
            f"\nEmin={min(energies):.8f}  Emax={max(energies):.8f}"
            f"\nmax|g|={maxg:.3e}"
            f"\nscore={score:.6f}"
        )

    return images

class DiagonalPreconditioner: # H ≈ diag(h), step Δx = -(H + λI)^-1 g
    def __init__(self, n_dim, h0 = 1.0):
        self.h = np.full(n_dim, h0)
        self.x_prev = None
        self.g_prev = None

    def update(self, x, g):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.g_prev = g.copy()
            return

        s = x - self.x_prev
        y = g - self.g_prev
        h_new = np.abs(y) / (np.abs(s) + EPS)

        self.h = 0.8 * self.h + 0.2 * h_new
        self.h = np.clip(self.h, 1e-4, 1e4)

        self.x_prev = x.copy()
        self.g_prev = g.copy()

    def step(self, g, damping = 1e-3):
        return -g / (self.h + damping)


# helpers
def reparameterize_linear(images):
    images = [np.asarray(x).copy() for x in images]
    n = len(images)

    s = np.zeros(len(images), dtype=float)

    for i in range(1, len(images)):
        ds = np.linalg.norm((images[i] - images[i - 1]).reshape(-1))
        s[i] = s[i - 1] + ds
    total = s[-1]

    s_old =  s / total if total > EPS else s
    s_new = np.linspace(0.0, 1.0, n)

    X = np.stack([x.reshape(-1) for x in images], axis=0)  # (n, ndof)
    ndof = X.shape[1]
    X_new = np.empty_like(X)

    for j in range(ndof):
        X_new[:, j] = np.interp(s_new, s_old, X[:, j])

    out = [X_new[i].reshape(images[0].shape) for i in range(n)]
    out[0], out[-1] = images[0].copy(), images[-1].copy()

    return out


def get_tangent(x_prev, x_next):
    return (x_next - x_prev).reshape(-1)

def get_unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + EPS)

def get_minus_cos(g, S):
    g = np.asarray(g).reshape(-1)
    S = np.asarray(S).reshape(-1)
    return -np.dot(g, S) / ((np.linalg.norm(g) + EPS) * (np.linalg.norm(S) + EPS))

def sum_chain_score(images, grads):
    score = 0.0
    for i in range(1, len(images) - 1):
        S = get_tangent(images[i - 1], images[i + 1])
        score += get_minus_cos(grads[i], S)
    return score



def compute_energies_grads(images: list[np.ndarray], atoms):
    energies, grads = [], []
    for x in images:
        E, grad = main_mopac_1d(atoms, x)
        energies.append(E)
        grads.append(np.asarray(grad))
    return energies, grads


def evolve_by_full_gradient(images,grads,
    dt: float = 0.05,
    trust_radius: float = 0.15,
    use_preconditioner: bool = True,
    damping: float = 1e-3,
    preconditioners: list[DiagonalPreconditioner] | None = None
) -> tuple[list[np.ndarray], list[DiagonalPreconditioner] | None]:
# φ̇_i = -∇V(φ_i)   (если grads = dE/dx)

    new_images = [images[0].copy()]

    if use_preconditioner:
        if preconditioners is None:
            preconditioners = [DiagonalPreconditioner(images[i].size) for i in range(1, len(images) - 1)]

    for i in range(1, len(images) - 1):
        x = np.asarray(images[i]).reshape(-1)

        g = np.asarray(grads[i]).reshape(-1)

        if use_preconditioner:
            pre = preconditioners[i - 1]
            pre.update(x, g)
            dx = dt * pre.step(g, damping=damping)
        else:
            dx = -dt * g

        # trust radius
        n = np.linalg.norm(dx)
        if n > trust_radius:
            dx *= trust_radius / (n + EPS)

        new_images.append((x + dx).reshape(images[i].shape))

    new_images.append(images[-1].copy())
    return new_images, preconditioners


def simplified_string_step(images,atoms,
    dt, trust_radius, use_preconditioner, damping,
    preconditioners: list[DiagonalPreconditioner] | None = None,
):
    energies, grads = compute_energies_grads(images, atoms)

    images, preconditioners = evolve_by_full_gradient(
        images,
        grads,
        dt=dt,
        trust_radius=trust_radius,
        use_preconditioner=use_preconditioner,
        damping=damping,
        preconditioners=preconditioners,
    )

    images = reparameterize_linear(images)
    return images, energies, grads, preconditioners