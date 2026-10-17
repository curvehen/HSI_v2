from scipy.stats import norm

def binomial_significance_test(n_trades, wins, baseline_p=0.5):
    if n_trades == 0:
        return None, None
    observed_p = wins / n_trades
    se = (baseline_p * (1 - baseline_p) / n_trades) ** 0.5
    z = (observed_p - baseline_p) / se
    p_value = 1 - norm.cdf(z)
    return round(z, 4), round(p_value, 6)
