# Функции для обработки данных

from typing import Tuple
import pandas as pd

from utils.constants import (
    TEST_MONTHS_COUNT,
    VALID_MONTHS_COUNT,
)


def get_splitted_data(
    time_series_data: pd.DataFrame, 
    test_months_count: int = TEST_MONTHS_COUNT,
    valid_months_count: int = VALID_MONTHS_COUNT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделяет данные таблицы на 3 части (тренировочные, тестовые и валидационные)

    Args:
        time_series_data (pd.DataFrame): Исходные данные
        test_months_count (int, optional): Количество месяцев, которые отделяем в тестовую выборку. 
            По умолчанию TEST_MONTHS_COUNT.
        valid_months_count(int, optional): Количество месяцев, которые отделяем в валидационную выборку. 
            По умолчанию VALID_MONTHS_COUNT.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            train_data - тренировочные данные, 
            test_data - тестовые данные
            valid_data - валидационные данные
    """
    rows_count = len(time_series_data)
    train_rows_count = rows_count - (test_months_count + valid_months_count)
    
    train_data = time_series_data.iloc[0:train_rows_count]
    test_data = time_series_data.iloc[train_rows_count:train_rows_count+test_months_count]
    valid_data = time_series_data.iloc[train_rows_count+test_months_count:rows_count]
    
    return train_data, test_data, valid_data