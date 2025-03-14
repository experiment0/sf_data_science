# Общие вспомогательные функции

import numpy as np
import pandas as pd
from sklearn import metrics
import datetime


def get_rmsle(y_pred_log: pd.Series, y_log: pd.Series) -> float:
    """Считает значение метрики RMSLE (Root Mean Squared Log Error)

    Args:
        y_pred_log (pd.Series): предсказание целевой переменной в логарифмическом масштабе
        y_log (pd.Series): значение целевой переменной, переведенное в логарифмический масштаб

    Returns:
        float: значение метрики RMSLE
    """
    mse = metrics.mean_squared_error(y_pred_log, y_log)
    return np.sqrt(mse)


def print_rmsle(rmsle: float, is_validation_data: bool = False):
    """Выводит значение метрики RMSLE

    Args:
        rmsle (float): значение метрики RMSLE
        is_validation_data (bool, optional): флаг, получена ли метрика на валидационной выборке. 
                                             По умолчанию False.
    """
    if (is_validation_data):
        print('RMSLE на валидационной выборке: ', rmsle)
    else:
        print('RMSLE на тренировочной выборке: ', rmsle)
        

def add_metric_to_data(metrics_data: pd.DataFrame, new_metric_data: list) -> pd.DataFrame:
    """Добавляет строку с новыми данными в таблицу с метриками

    Args:
        metrics_data (pd.DataFrame): таблица с метриками
        new_metric_data (list): новые данные

    Returns:
        pd.DataFrame: таблица с добавленной строкой
    """
    metrics_data = metrics_data.copy()
    metrics_data.loc[len(metrics_data.index)] = new_metric_data
    
    return metrics_data
    

def get_multicolinear_pairs(data: pd.DataFrame, corr_coeff: float, method: str = 'pearson') -> list:
    """Возвращает пары мультиколинеарных признаков и значение кореляции между ними.
    Args:
        data (pd.DataFrame): таблица с данными
        corr_coeff (float): коэффициент корреляции, выше которого отбираем пары
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
    Returns:
        list: Список с именами столбцов и коэффициентом корреляции
              ([столбец1, столбец2], коэффициент)
    """
    # Матрица корреляции признаков
    data_corr = data.corr(method=method)
    
    # Имена колонок таблицы
    col_names = list(data.columns)
    
    # Соберем в этот список пары колонок без повторений
    colls_pairs = []
    
    for col1 in col_names:
        for col2 in col_names:
            if (col1 != col2 and [col1, col2] not in colls_pairs and [col2, col1] not in colls_pairs):
                colls_pairs.append([col1, col2])
    
    # Соберем в этот список пары признаков, 
    # у которых коэффициент корреляции больше corr_coeff
    multicolinear_pairs = []
    
    # Перебираем все пары признаков без повторений
    for pair in colls_pairs:
        # Вынимаем коэффициент корреляции признаков из матрицы корреляции
        corr_between_colls = round(data_corr[pair[0]][pair[1]], 2)
        # Если по модулю коэффициент корреляции больше
        if (abs(corr_between_colls) > abs(corr_coeff)):
            # Добавляем пару в наш список
            multicolinear_pairs.append((pair, corr_between_colls))
            
    return multicolinear_pairs


def get_minutes(seconds: int) -> float:
    """Переводит секунды в минуты

    Args:
        seconds (int): количество секунд

    Returns:
        float: количество минут
    """
    return round(seconds / 60, 2)


def get_days(seconds: int) -> float:
    """Переводит секунды в дни

    Args:
        seconds (int): количество секунд

    Returns:
        float: количество дней
    """
    return round(seconds/(60*60*24), 2)


def print_lead_time(lead_time: float):
    """Выводит время выполнения ячейки

    Args:
        lead_time (float): время выполнения в секундах
    """
    print('Время выполнения: ', str(datetime.timedelta(seconds=lead_time)))
    
    