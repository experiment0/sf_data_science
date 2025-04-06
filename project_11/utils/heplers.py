# Содержит различные вспомогательные функции

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller


def check_time_series_stationarity(time_series: pd.DataFrame) -> None:
    """Определяет, является ли ряд стационарным для уровня значимости 5%
    и выводит информацию о результатах проверки.

    Args:
        time_series (pd.DataFrame): временной ряд
    """
    test_result = adfuller(time_series.values)
    # Значение статистики из теста
    adf_statistic = test_result[0]
    # Значение p-value
    pvalue = test_result[1]
    # Критические значения разных уровней значимости (1%, 5%, 10%)
    critical_values = test_result[4]
    
    print('Значение статистики теста:', adf_statistic)
    print('p-value:', pvalue)
    print('Критические значения разных уровней значимости:')
    for level, value in critical_values.items():
        print(f'    {level}: {value}')

    # Если значение статистики ниже выбранного уровня значимости,
    if (adf_statistic < critical_values['5%']):
        # Ряд стационарный
        print(f'Значение статистики {adf_statistic} меньше выбранного уровня значимости 5%.')
        print('Отвергаем нулевую гипотезу, РЯД СТАЦИОНАРНЫЙ.')
    else:
        print(f'Значение статистики {adf_statistic} больше выбранного уровня значимости значимости 5%.')
        print('У нас нет оснований отвергнуть нулевую гипотезу, РЯД НЕ СТАЦИОНАРНЫЙ.')


def get_mape_in_percent(true_values: pd.Series, predict_values: pd.Series) -> float:
    """Считает метрику MAPE и переводит ее в проценты и округляет до сотых

    Args:
        true_values (pd.Series): истинные значения
        predict_values (pd.Series): предсказанные значения

    Returns:
        float: метрика MAPE, переведенная в проценты и округленная до сотых
    """
    mape = mean_absolute_percentage_error(true_values, predict_values)
    return round(mape*100, 2)


def display_mape(true_values: pd.Series, predict_values: pd.Series) -> None:
    """Печатает значение метрики MAPE

    Args:
        true_values (pd.Series): истинные значения
        predict_values (pd.Series): предсказанные значения
    """
    mape_in_percent = get_mape_in_percent(true_values, predict_values)
    print(f'Метрика MAPE: {mape_in_percent} %')