"""
Cox proportional hazards regression, Efron tie handling.

A small, dependency-free (numpy only) re-implementation of the defaults of
R's survival::coxph(): Newton-Raphson with step halving, Efron ties,
convergence on relative change of the log partial likelihood (eps = 1e-9).
Prediction uses R's convention of centring covariates on the training means:
    risk = exp((x - mean_train) . beta)          [predict(..., type = "risk")]
"""
import numpy as np


def _efron(beta, X, time, event):
    """Log partial likelihood, gradient and Hessian (Efron ties)."""
    eta = X @ beta
    eta_max = eta.max()
    w = np.exp(eta - eta_max)                       # scaled for numerical safety
    order = np.argsort(-time, kind="mergesort")     # descending time
    t, d, Xo, wo, etao = time[order], event[order], X[order], w[order], eta[order]
    p = X.shape[1]

    # Running risk-set sums (all subjects with time >= current time)
    S0 = np.cumsum(wo)
    S1 = np.cumsum(wo[:, None] * Xo, axis=0)
    S2 = np.cumsum(wo[:, None, None] * Xo[:, :, None] * Xo[:, None, :], axis=0)

    ll, grad, hess = 0.0, np.zeros(p), np.zeros((p, p))
    n = len(t)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        # block i..j shares one time; risk set = rows 0..j in descending order
        dead = np.where(d[i:j + 1] == 1)[0] + i
        m = len(dead)
        if m:
            r0, r1, r2 = S0[j], S1[j], S2[j]
            e0 = wo[dead].sum()
            e1 = (wo[dead][:, None] * Xo[dead]).sum(axis=0)
            e2 = (wo[dead][:, None, None] * Xo[dead][:, :, None] * Xo[dead][:, None, :]).sum(axis=0)
            ll += etao[dead].sum()
            grad += Xo[dead].sum(axis=0)
            for k in range(m):
                f = k / m
                a0 = r0 - f * e0
                a1 = r1 - f * e1
                a2 = r2 - f * e2
                ll -= np.log(a0) + eta_max
                mu = a1 / a0
                grad -= mu
                hess -= a2 / a0 - np.outer(mu, mu)
        i = j + 1
    return ll, grad, hess


class CoxPH:
    def __init__(self, eps=1e-9, max_iter=50):
        self.eps, self.max_iter = eps, max_iter

    def fit(self, X, time, event, names=None):
        X = np.asarray(X, float)
        time = np.asarray(time, float)
        event = np.asarray(event, float)
        ok = ~(np.isnan(X).any(axis=1) | np.isnan(time) | np.isnan(event))
        X, time, event = X[ok], time[ok], event[ok]

        self.means_ = X.mean(axis=0)
        Xc = X - self.means_                     # centring as R does (stability)
        beta = np.zeros(X.shape[1])
        ll, g, H = _efron(beta, Xc, time, event)
        self.loglik_null_ = ll
        for it in range(self.max_iter):
            step = np.linalg.solve(-H, g)
            new = beta + step
            ll_new, g_new, H_new = _efron(new, Xc, time, event)
            halv = 0
            while ll_new < ll and halv < 30:         # step halving
                step /= 2
                new = beta + step
                ll_new, g_new, H_new = _efron(new, Xc, time, event)
                halv += 1
            converged = abs(1 - ll / ll_new) <= self.eps
            beta, ll, g, H = new, ll_new, g_new, H_new
            if converged:
                break
        self.n_iter_ = it + 1
        self.coef_ = beta
        self.loglik_ = ll
        self.var_ = np.linalg.inv(-H)
        self.se_ = np.sqrt(np.diag(self.var_))
        self.names_ = list(names) if names is not None else [f"x{i}" for i in range(len(beta))]
        self.n_, self.n_events_ = len(time), int(event.sum())
        return self

    def linear_predictor(self, X):
        return (np.asarray(X, float) - self.means_) @ self.coef_

    def predict_risk(self, X):
        return np.exp(self.linear_predictor(X))

    def summary(self):
        from math import erf, sqrt
        z = self.coef_ / self.se_
        p = [2 * (1 - 0.5 * (1 + erf(abs(v) / sqrt(2)))) for v in z]
        return [(n, b, np.exp(b), s, zz, pp)
                for n, b, s, zz, pp in zip(self.names_, self.coef_, self.se_, z, p)]
