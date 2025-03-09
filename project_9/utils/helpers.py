# Различные вспомогательные функции

import numpy as np
from numpy.linalg import norm
import pandas as pd
import time


def get_multicolinear_pairs(
    data: pd.DataFrame, 
    corr_coeff: float = 0.7, 
    method: str = 'pearson',
) -> list:
    """Возвращает пары мультиколинеарных признаков и значение кореляции между ними.
    Args:
        data (pd.DataFrame): таблица с данными
        corr_coeff (float): коэффициент корреляции, выше которого отбираем пары. По умолчанию 0.7
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
    Returns:
        list: Список с именами столбцов и коэффициентом корреляции
              ([столбец1, столбец2], коэффициент)
    """
    # Матрица корреляции признаков
    data_corr = data.select_dtypes(include=np.number).corr(method=method)
        
    # Имена колонок таблицы
    col_names = list(data_corr.columns)
    
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


def get_exec_time(start: float, end: float) -> str:
    """Возвращает время выполнения в секундах

    Args:
        start (float): время начала
        end (float): время окончания

    Returns:
        str: строка с форматированным временем выполнения
    """
    duration = end - start
    formatted_duration = time.strftime("%H:%M:%S", time.gmtime(duration))
    return formatted_duration


def get_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Считает косинус угла между двумя векторами

    Args:
        A (np.ndarray): первый вектор
        B (np.ndarray): второй вектор

    Returns:
        float: косинус угла между векторами
    """
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    return cosine