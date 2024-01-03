# Содержит функции для преобразования данных

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

from functions.constants import (
    CURRENT_YEAR, TARGET_FEATURE, RANDOM_STATE, TEST_SIZE
)


def get_balance(value: str) -> float:
    """Переводит признак баланса в числовой

    Args:
        value (str): строковое представление баланса

    Returns:
        float: значение баланса, преобразованное в число
    """
    # Пропуски пока оставим нетронутыми
    if (pd.isna(value)):
        return np.nan
    
    # Выделим из строки числовое значение баланса
    balance = value.replace('$', '').replace(' ', '').replace(',', '.')
    
    return float(balance)


def get_data_without_outliers(data_source: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Удаляет выбросы по методу Тьюки

    Args:
        data_source (pd.DataFrame): таблица с данными
        feature (str): признак, для которого нужно обработать выбросы

    Returns:
        pd.DataFrame: таблица после удаления выбросов для переданного признака
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    # Значения признака
    x = data[feature]
    
    # Вычисляем 25 и 75 квантили
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75)
    
    # Вычисляем межквартильное расстояние
    iqr = quartile_3 - quartile_1
    
    # Вычисляем верхнюю и нижнюю границу Тьюки
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    
    # Маска для определения выбросов
    outliers_mask = (x < lower_bound) | (x > upper_bound)
    
    # Удаляем выбросы по маске
    cleaned_data = data.drop(index=data[outliers_mask].index, axis=0)
    
    return cleaned_data


def get_data_with_label_encoded_columns(data_source: pd.DataFrame) -> pd.DataFrame:
    """Перекодирует категориальные признаки с помощью LabelEncoder,
    чтобы можно было построить матрицу корреляции

    Args:
        data_source (pd.DataFrame): таблица с данными

    Returns:
        pd.DataFrame: таблица с перекодированными столбцами
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    LE = LabelEncoder()
    
    # Выделим столбцы с типом object, все они содержат категориальные признаки
    categorial_fields = data.select_dtypes(include=['object']).columns
    
    # Перекодируем признаки
    for field_name in categorial_fields:
        data[field_name] = LE.fit_transform(data[field_name])
        
    return data


def get_season(month: str) -> str:
    """Возвращает название времени года, к которому относится переданный месяц

    Args:
        month (str): обозначение месяца

    Returns:
        str: название времени года, к которому относится переданный месяц
    """
    if (month in ['dec', 'jan', 'feb']):
        return 'winter'
    if (month in ['mar', 'apr', 'may']):
        return 'spring'
    if (month in ['jun', 'jul', 'aug']):
        return 'summer'
    return 'autumn'


# Словарь соответствия месяцев цифрам
month_to_number_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 
    'apr': 4, 'may': 5, 'jun': 6, 
    'jul': 7, 'aug': 8, 'sep': 9, 
    'oct': 10, 'nov': 11, 'dec': 12,
}


# Формируем дату из месяца и дня
def get_date(month: str, day: int) -> datetime:
    """Формирует дату из переданного месяца и дня (для текущего года)

    Args:
        month (str): обозначение месяца
        day (int): номер дня в месяце

    Returns:
        datetime: дата
    """
    month_number = month_to_number_map[month]
    date = datetime(CURRENT_YEAR, month_number, day)
    return date


def get_feature_scale(
        feature_value: Union[int, float], 
        thresholds: range
    ) -> int:
    """Возвращает индекс диапазона, в котором находится значение признака.

    Args:
        feature_value (Union[int, float]): значение признака
        thresholds (range): шкала с диапазонами

    Returns:
        int: индекс шкалы диапазона, в котором находится значение признака
    """
    for index, threshold in enumerate(thresholds):
        if (feature_value <= threshold):
            return index
        
    return len(list(thresholds))


def get_education_range(value):
    if (value == 'primary'):
        return 1
    if (value == 'secondary'):
        return 2
    return 3


def get_samples(data_source: pd.DataFrame):
    """Формирует тренировочную и тестовую выборки

    Args:
        data_source (pd.DataFrame): исходная таблица

    Returns:
        тренировочную и тестовую выборки
    """
    data = data_source.copy()
    
    X = data.drop([TARGET_FEATURE], axis=1)
    y = data[TARGET_FEATURE]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, stratify=y, random_state=RANDOM_STATE, test_size=TEST_SIZE)
    
    return X_train, X_test, y_train, y_test
