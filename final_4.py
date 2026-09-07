import pandas as pd
import numpy as np

# ---------- Загрузка данных ----------
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')  # замените на ваш файл
df = df.sort_values('fico_score').reset_index(drop=True)

fico = df['fico_score'].values
defaults = df['default'].values

# ---------- Подход 1: минимизация MSE (быстрый, интерпретируемый) ----------
def mse_quantization(fico_sorted, n_buckets):
    """
    Разбиение отсортированного массива FICO на n_buckets бакетов,
    минимизирующее сумму квадратичных отклонений от среднего внутри бакета.
    Решается точным ДП (как k-means для 1D, оптимально в отличие от обычного k-means).
    """
    n = len(fico_sorted)
    # Префиксные суммы для быстрого счёта суммы и суммы квадратов на отрезке
    prefix_sum = np.concatenate([[0], np.cumsum(fico_sorted)])
    prefix_sq = np.concatenate([[0], np.cumsum(fico_sorted ** 2)])

    def segment_cost(i, j):
        # стоимость (сумма квадратов отклонений от среднего) отрезка [i, j)
        count = j - i
        s = prefix_sum[j] - prefix_sum[i]
        sq = prefix_sq[j] - prefix_sq[i]
        mean = s / count
        return sq - count * mean ** 2

    # dp[k][i] = минимальная стоимость разбиения первых i элементов на k бакетов
    dp = np.full((n_buckets + 1, n + 1), np.inf)
    split = np.zeros((n_buckets + 1, n + 1), dtype=int)
    dp[0][0] = 0

    for k in range(1, n_buckets + 1):
        for i in range(1, n + 1):
            for j in range(k - 1, i):
                cost = dp[k - 1][j] + segment_cost(j, i)
                if cost < dp[k][i]:
                    dp[k][i] = cost
                    split[k][i] = j

    # восстанавливаем границы
    boundaries_idx = []
    idx = n
    for k in range(n_buckets, 0, -1):
        j = split[k][idx]
        boundaries_idx.append(j)
        idx = j
    boundaries_idx = sorted(boundaries_idx)

    boundaries = [fico_sorted[i] for i in boundaries_idx[1:]]  # без первой (это 0)
    return boundaries, dp[n_buckets][n]


# ---------- Подход 2: максимизация log-likelihood (ДП, учитывает дефолты) ----------
def log_likelihood_quantization(fico_sorted, default_sorted, n_buckets):
    """
    ДП по log-likelihood из задания:
    LL = sum_i [ k_i * log(p_i) + (n_i - k_i) * log(1 - p_i) ],
    где p_i = k_i / n_i — доля дефолтов в бакете i.
    """
    n = len(fico_sorted)
    prefix_defaults = np.concatenate([[0], np.cumsum(default_sorted)])

    def segment_ll(i, j):
        # log-likelihood отрезка [i, j)
        count = j - i
        k = prefix_defaults[j] - prefix_defaults[i]
        if count == 0:
            return 0.0
        p = k / count
        if p == 0 or p == 1:
            # log(0) не определён — вклад отрезка с чистым 0/1 считаем нулевым лог-риском
            return 0.0
        return k * np.log(p) + (count - k) * np.log(1 - p)

    NEG_INF = -np.inf
    dp = np.full((n_buckets + 1, n + 1), NEG_INF)
    split = np.zeros((n_buckets + 1, n + 1), dtype=int)
    dp[0][0] = 0.0

    for k in range(1, n_buckets + 1):
        for i in range(1, n + 1):
            for j in range(k - 1, i):
                if dp[k - 1][j] == NEG_INF:
                    continue
                val = dp[k - 1][j] + segment_ll(j, i)
                if val > dp[k][i]:
                    dp[k][i] = val
                    split[k][i] = j

    boundaries_idx = []
    idx = n
    for k in range(n_buckets, 0, -1):
        j = split[k][idx]
        boundaries_idx.append(j)
        idx = j
    boundaries_idx = sorted(boundaries_idx)

    boundaries = [fico_sorted[i] for i in boundaries_idx[1:]]
    return boundaries, dp[n_buckets][n]


# ---------- Построение rating map ----------
def build_rating_map(boundaries, higher_score_is_better=True):
    """
    boundaries — границы между бакетами (отсортированные значения FICO).
    Возвращает функцию fico_score -> rating,
    где rating=0 (или 1, по вкусу) соответствует ЛУЧШЕМУ кредитному качеству,
    как требуется в задании ("lower rating = better score").
    """
    boundaries = sorted(boundaries)

    def fico_to_rating(fico_score):
        bucket = np.searchsorted(boundaries, fico_score, side='right')
        n_buckets = len(boundaries) + 1
        if higher_score_is_better:
            # чем выше FICO, тем ниже (лучше) рейтинг
            rating = (n_buckets - 1) - bucket
        else:
            rating = bucket
        return rating

    return fico_to_rating


# ---------- Запуск на данных ----------
n_buckets = 5

boundaries_mse, mse_cost = mse_quantization(fico, n_buckets)
print(f"Границы по MSE: {boundaries_mse}")

boundaries_ll, ll_value = log_likelihood_quantization(fico, defaults, n_buckets)
print(f"Границы по log-likelihood: {boundaries_ll}, LL = {ll_value:.2f}")

rating_map = build_rating_map(boundaries_ll)

# тест на примерах
for score in [580, 605, 650, 700, 740, 800]:
    print(f"FICO {score} -> rating {rating_map(score)}")