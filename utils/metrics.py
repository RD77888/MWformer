import numpy as np

def _flatten_and_mask(pred, true):

    pred = np.asarray(pred)
    true = np.asarray(true)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return np.array([]), np.array([])
    return pred[mask].ravel(), true[mask].ravel()


def RSE(pred, true):
    pred_f, true_f = _flatten_and_mask(pred, true)
    if true_f.size == 0:
        return np.nan
    num = np.sum((true_f - pred_f) ** 2)
    den = np.sum((true_f - true_f.mean()) ** 2)
    if den == 0:
        return np.nan
    return np.sqrt(num) / np.sqrt(den)


def R2_score(pred, true):
    pred_f, true_f = _flatten_and_mask(pred, true)
    if true_f.size == 0:
        return np.nan
    num = np.sum((true_f - pred_f) ** 2)
    den = np.sum((true_f - true_f.mean()) ** 2)
    if den == 0:
        return np.nan
    return 1 - num / den


def CORR(pred, true):
    pred_f, true_f = _flatten_and_mask(pred, true)
    if true_f.size == 0:
        return np.nan
    x = true_f - true_f.mean()
    y = pred_f - pred_f.mean()
    num = np.sum(x * y)
    den = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if den == 0:
        return np.nan
    return num / den


def MAE(pred, true):
    pred_f, true_f = _flatten_and_mask(pred, true)
    if true_f.size == 0:
        return np.nan
    return np.mean(np.abs(pred_f - true_f))


def MSE(pred, true):
    pred_f, true_f = _flatten_and_mask(pred, true)
    if true_f.size == 0:
        return np.nan
    return np.mean((pred_f - true_f) ** 2)


def RMSE(pred, true):
    val = MSE(pred, true)
    return np.sqrt(val) if np.isfinite(val) else np.nan


def MAPE(pred, true, eps: float = 1e-8):
    pred_f, true_f = _flatten_and_mask(pred, true)
    if true_f.size == 0:
        return np.nan
    denom = np.where(np.abs(true_f) < eps, eps, np.abs(true_f))
    return np.mean(np.abs(pred_f - true_f) / denom)


def MSPE(pred, true, eps: float = 1e-8):
    pred_f, true_f = _flatten_and_mask(pred, true)
    if true_f.size == 0:
        return np.nan
    denom = np.where(np.abs(true_f) < eps, eps, np.abs(true_f))
    return np.mean(((pred_f - true_f) / denom) ** 2)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr
