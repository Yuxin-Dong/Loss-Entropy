import numpy as np
import torch
from scipy import stats
import scipy.optimize as optimize
import scipy.special as special

from nnlib.nnlib import utils
from methods import LangevinDynamics


def mi(prob):
    px = np.sum(prob, axis=2)
    py = np.sum(prob, axis=1)
    mi = np.sum(prob * np.log(prob / px[:, :, None] / py[:, None, :]), axis=(1, 2))
    return np.maximum(0, mi)


def entr(prob, alpha=1.0):
    prob = np.reshape(prob, (prob.shape[0], -1))
    if alpha == 1.0:
        res = -np.sum(prob * np.log(prob), axis=1)
    else:
        res = np.log(np.sum(np.power(prob, alpha), axis=1)) / (1 - alpha)
    return np.maximum(0, res)


def disc(xs, nx=2):
    prob = np.zeros((xs.shape[1], nx))
    np.add.at(prob, (np.arange(xs.shape[1])[None, :], xs), 1.0 / xs.shape[0])
    prob = np.clip(prob, 1e-9, None)
    return prob


def disc2(xs, nx, ys, ny=2):
    prob = np.zeros((xs.shape[1], nx, ny))
    np.add.at(prob, (np.arange(xs.shape[1])[None, :], xs, ys), 1.0 / xs.shape[0])
    prob = np.clip(prob, 1e-9, None)
    return prob


def disc3(xs, nx, ys, ny, zs, nz=2):
    prob = np.zeros((xs.shape[1], nx, ny, nz))
    np.add.at(prob, (np.arange(xs.shape[1])[None, :], xs, ys, zs), 1.0 / xs.shape[0])
    prob = np.clip(prob, 1e-9, None)
    return prob


def bin(xs, bw=1):
    mn, mx = np.min(xs), np.max(xs)
    if mn < -1e-9:
        mn = (-np.ceil((-bw / 2 - mn) / bw) - 0.5) * bw
    nb = int(np.ceil((mx - mn) / bw))
    bs = np.clip(((xs - mn) / bw).astype(np.int32), 0, nb - 1)
    return bs, nb


def disc_entr(xs, nx=2, alpha=1.0):
    return entr(disc(xs, nx), alpha)


def disc2_entr(xs, nx, ys, ny=2, alpha=1.0):
    return entr(disc2(xs, nx, ys, ny), alpha)


def disc_mi(xs, nx, ys, ny=2):
    return mi(disc2(xs, nx, ys, ny))


def disc3_mi(xs, nx, ys, ny, zs, nz=2):
    prob = disc3(xs, nx, ys, ny, zs, nz)
    prob = np.reshape(prob, (prob.shape[0], -1, prob.shape[3]))
    return mi(prob)


def bin_prob(xs, bw=1):
    return disc(*bin(xs, bw))


def bin2_prob(xs, ys, bw=1):
    return disc2(*bin(xs, bw), *bin(ys, bw))


def bin_disc_prob(xs, ys, ny=2, bw=1):
    return disc2(*bin(xs, bw), ys, ny)


def bin_entr(xs, bw=1, alpha=1.0):
    return entr(disc(*bin(xs, bw)), alpha)


def bin2_entr(xs, ys, bw=1, alpha=1.0):
    return entr(disc2(*bin(xs, bw), *bin(ys, bw)), alpha)


def bin_disc_entr(xs, ys, ny=2, bw=1, alpha=1.0):
    return entr(disc2(*bin(xs, bw), ys, ny), alpha)


def bin_mi(xs, ys, bw=1):
    return mi(disc2(*bin(xs, bw), *bin(ys, bw)))


def bin_disc_mi(xs, ys, ny=2, bw=1):
    return mi(disc2(*bin(xs, bw), ys, ny))


def bin2_disc_mi(xs, ys, zs, nz=2, bw=1):
    prob = disc3(*bin(xs, bw), *bin(ys, bw), zs, nz)
    prob = np.reshape(prob, (prob.shape[0], -1, prob.shape[3]))
    return mi(prob)


def estimate_fcmi_bound_classification(masks, preds, num_examples, num_classes,
                                       verbose=False, return_list_of_mis=False):
    bound = 0.0
    list_of_mis = []
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2 * idx:2 * idx + 2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = disc_mi(ms, 2, ps, num_classes ** 2)
        list_of_mis.append(cur_mi)
        bound += np.sqrt(2 * cur_mi)
        if verbose and idx < 10:
            print("ms:", ms)
            print("ps:", ps)
            print("mi:", cur_mi)
    bound *= 1 / num_examples

    if return_list_of_mis:
        return bound, list_of_mis

    return bound


def minimize_lambda(prob, delta):
    def func(x):
        return np.sum(entr(prob, 1 - x)) + delta / x

    def grad(x):
        prob_alpha = np.power(prob, 1 - x)
        prob_alpha /= np.sum(prob_alpha, axis=1)[:, None]
        return (np.sum(prob_alpha * np.log(prob_alpha / prob)) - delta) / x ** 2

    res = optimize.minimize(func, np.array([0.05]), jac=grad, method="L-BFGS-B", bounds=((1e-9, 1.0 - 1e-9),))

    return res.fun


def estimate_subgauss(losses):
    def func(x):
        return -special.logsumexp(losses * x, b=1 / losses.shape[0]) * 2 / x ** 2

    def grad(x):
        sexp = np.exp(losses * x)
        return special.logsumexp(losses * x, b=1 / losses.shape[0]) * 4 / x ** 3\
            - 2 * np.mean(losses * sexp) / np.mean(sexp) / x ** 2

    res = optimize.minimize(func, np.array([1]), jac=grad, method="L-BFGS-B", bounds=[(1e-3, None)])

    return -res.fun


def estimate_loo_bound(masks, losses, n, delta=0.1):
    masks = np.array(masks)
    losses = np.array(losses)

    loss_mask = np.ones(losses.shape, dtype=bool)
    loss_mask[np.arange(masks.shape[0]), masks] = False
    train_losses = np.reshape(losses[loss_mask], (masks.shape[0], n))
    val_losses = losses[np.arange(masks.shape[0]), masks]

    loss_mean = (n + 1) / n * (losses - np.mean(losses, axis=1)[:, None])

    delta_prob = bin_prob(np.concatenate([train_losses, val_losses[:, None]], axis=1))
    delta_entr = minimize_lambda(delta_prob, np.log(1 / delta))
    loo_bound = 2 * np.array([estimate_subgauss(loss_mean[i, :]) for i in range(losses.shape[0])])
    loo_bound *= delta_entr + np.log(2 / delta)
    loo_bound = np.sqrt(loo_bound)

    return loo_bound


def minimize_weighted(n, delta, pair_prob, loss_train, loss_max):
    pair_entr = minimize_lambda(pair_prob, np.log(1 / delta)) / n
    weighted_bound = pair_entr + np.log(4 / delta) / n
    weighted_bound *= 2 * np.max(loss_max, axis=1) / np.log(2)

    if np.max(loss_train) < 1e-9:
        return weighted_bound, weighted_bound
    else:
        def func(x):
            eta = x * np.log(2) / 2 / np.max(loss_max, axis=1)
            c = -np.log(2 - np.exp(2 * eta[:, None] * loss_max)) / (2 * eta[:, None] * loss_max) - 1
            C = np.max(c)

            return weighted_bound / x + np.mean(c * loss_train, axis=1), weighted_bound / x + np.mean(C * loss_train, axis=1)

        res = optimize.minimize(lambda x: np.mean(func(x)[0]), np.array([0.5]), method="Nelder-Mead", bounds=((1e-9, 1.0 - 1e-9),))
        return func(res.x)


def minimize_fastrate(n, delta, loss_0, loss_1, loss_train, loss_max):
    if np.max(loss_train) < 1e-9:
        def func(x):
            fast_bound_1 = np.mean(entr(bin2_prob(np.minimum(loss_0, x[0]), np.minimum(loss_1, x[0])), 1 - x[1]))
            fast_bound_1 += (np.log(2 / delta) / x[1] + np.log(8 / delta)) / n
            fast_bound_1 *= 2 * x[0] / np.log(2)

            loss_delta = np.maximum(loss_1 - x[0], 0) - np.maximum(loss_0 - x[0], 0)
            fast_bound_2 = np.mean(entr(bin_prob(loss_delta / 2), 1 - x[2]))
            fast_bound_2 += (np.log(2 / delta) / x[2] + np.log(4 / delta)) / n
            fast_bound_2 *= 2 * np.mean(loss_delta ** 2, axis=1)
            fast_bound_2 = np.sqrt(fast_bound_2)

            print(x, np.mean(fast_bound_1), np.mean(fast_bound_2), np.mean(fast_bound_1 + fast_bound_2))
            return fast_bound_1 + fast_bound_2

        res = optimize.minimize(lambda x: np.mean(func(x)), np.array([1.0, 0.5, 0.5]), method="Nelder-Mead",
                                bounds=((1e-9, 10.0), (1e-9, 1.0 - 1e-9), (1e-9, 1.0 - 1e-9)))

    else:
        def func(x):
            fast_bound_1 = np.mean(entr(bin2_prob(np.minimum(loss_0, x[0]), np.minimum(loss_1, x[0])), 1 - x[1]))
            fast_bound_1 += (np.log(2 / delta) / x[1] + np.log(8 / delta)) / n
            fast_bound_1 *= 2 * x[0] / x[3] / np.log(2)

            loss_delta = np.maximum(loss_1 - x[0], 0) - np.maximum(loss_0 - x[0], 0)
            fast_bound_2 = np.mean(entr(bin_prob(loss_delta / 2), 1 - x[2]))
            fast_bound_2 += (np.log(2 / delta) / x[2] + np.log(4 / delta)) / n
            fast_bound_2 *= 2 * np.mean(loss_delta ** 2, axis=1)
            fast_bound_2 = np.sqrt(fast_bound_2)

            loss_max_kappa = x[3] * np.log(2) * np.minimum(loss_max, x[0]) / x[0]
            c = -np.log(2 - np.exp(loss_max_kappa)) / loss_max_kappa - 1
            fast_bound_3 = np.mean(c * np.minimum(loss_train, x[0]), axis=1)

            print(x, np.mean(fast_bound_1), np.mean(fast_bound_2), np.mean(fast_bound_3), np.mean(fast_bound_1 + fast_bound_2 + fast_bound_3))
            return fast_bound_1 + fast_bound_2 + fast_bound_3

        res = optimize.minimize(lambda x: np.mean(func(x)), np.array([1.0, 0.5, 0.5, 0.5]), method="Nelder-Mead",
                                bounds=((1e-9, 10.0), (1e-9, 1.0 - 1e-9), (1e-9, 1.0 - 1e-9), (1e-9, 1.0 - 1e-9)))

    return func(res.x)


def binary_kl_bound(q, c):
    q = max(q, 1e-9)

    def func(x):
        return q * np.log(2 * q / (q + x)) + (1 - q) * np.log((1 - q) / (1 - (q + x) / 2)) - c

    res = optimize.root_scalar(func, bracket=[q, 1], method='brentq')

    return res.root

def estimate_ss_bound(masks, losses, n, delta=0.5):
    masks = np.array(masks)
    losses = np.array(losses)

    loss_0 = losses[:, ::2]
    loss_1 = losses[:, 1::2]
    loss_pair = np.stack([loss_0, loss_1], axis=2)
    loss_delta = loss_1 - loss_0
    loss_train = loss_pair[np.arange(masks.shape[0])[:, None], np.arange(masks.shape[1])[None, :], masks]
    # loss_val = loss_pair[np.arange(masks.shape[0])[:, None], np.arange(masks.shape[1])[None, :], 1 - masks]
    loss_max = np.max(loss_pair, axis=2) + 1e-9

    delta_prob = bin_prob(loss_delta / 2)
    pair_prob = bin2_prob(loss_0, loss_1)
    pair_mi = bin2_disc_mi(loss_0, loss_1, masks)

    delta_entr = minimize_lambda(delta_prob, np.log(1 / delta)) / n
    ss_bound = delta_entr + np.log(2 / delta) / n
    ss_bound *= 2 * np.mean(loss_delta ** 2, axis=1)
    ss_bound = np.sqrt(ss_bound)

    fast_bound = minimize_fastrate(n, delta, loss_0, loss_1, loss_train, loss_max)

    weighted_bound, weighted_bound_c = minimize_weighted(n, delta, pair_prob, loss_train, loss_max)

    binary_bound = binary_kl_bound(np.mean(loss_train) / np.max(loss_max), np.mean(pair_mi) + np.log(2 * np.sqrt(n) / delta) / n) * np.max(loss_max)

    return ss_bound, fast_bound, weighted_bound, weighted_bound_c, binary_bound


def estimate_sgld_bound(n, batch_size, model):
    """ Computes the bound of Negrea et al. "Information-Theoretic Generalization Bounds for
    SGLD via Data-Dependent Estimates". Eq (6) of https://arxiv.org/pdf/1911.02151.pdf.
    """
    assert isinstance(model, LangevinDynamics)
    assert model.track_grad_variance
    T = len(model._grad_variance_hist)
    assert len(model._lr_hist) == T + 1
    assert len(model._beta_hist) == T + 1
    ret = 0.0
    for t in range(1, T):  # skipping the first iteration as grad_variance was not tracked for it
        ret += model._lr_hist[t] * model._beta_hist[t] / 4.0 * model._grad_variance_hist[t - 1]
    ret = np.sqrt(utils.to_numpy(ret))
    ret *= np.sqrt(n / 4.0 / batch_size / (n - 1) / (n - 1))
    return ret
