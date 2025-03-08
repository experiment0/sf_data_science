# Функции для обработки и преобразования данных

from typing import Tuple
import numpy as np
import pandas as pd

from utils.constants import (
    F,
    form_fields,
)

# Данные за прошлые года (нужны для построения графика на этапе прогноза)
past_data = pd.read_csv('./data/past_data.csv')

# Данные за года, по которым будет строится прогноз. 
# Посчитаны путем экспоненциального сглаживания предыдущих значений.
# Либо взято значение константы, определенное для прошлых лет (для NegativeCoef)
future_data = pd.read_csv('./data/future_data.csv')


def get_locations() -> dict:
    """Возвращает словарь с данными стран для вывода в форме выбора страны на первом экране

    Returns:
        dict: словарь с данными стран в формате код_страны: название_страны
    """
    locations_dict = past_data\
    .groupby(F.SpatialDimValueCode.value)\
    .first()[[F.Location.value]]\
    .to_dict()[F.Location.value]
    
    return locations_dict


def get_location_by_code(code: str) -> str:
    """Возвращает название страны по ее коду

    Args:
        code (str): код страны

    Returns:
        str: название страны
    """
    locations = get_locations()
    
    return locations[code]


def get_location_data(data: pd.DataFrame, code: str) -> pd.DataFrame:
    """Возвращает данные страны для переданного кода

    Args:
        data (pd.DataFrame): таблица, из которой нужно извлечь данные страны
        code (str): код страны

    Returns:
        pd.DataFrame: данные нужной страны
    """
    mask = data[F.SpatialDimValueCode.value] == code
    location_data = data[mask].reset_index(drop=True)
    
    return location_data


def get_form_predictors(code: str) -> Tuple[np.ndarray, dict]:
    """Возвращает данные по стране для вывода в форме задания предикторов (вторая страница)

    Args:
        code (str): код страны

    Returns:
        Tuple[np.ndarray, dict]: 
            - список лет, за которые будет делаться прогноз
            - словарь со сглаженными значениями предыдущих лет, 
            которые пользователь сможет поменять в форме
    """
    # Предполагаемые данные за будущие года для страны
    location_future_data = get_location_data(future_data, code)
    # Года, за которые будет делаться прогноз    
    form_years = location_future_data[F.Period.value].values
    # Словарь со сглаженными значениями для вывода в форме
    form_predictors = location_future_data[
        [F.Period.value] + form_fields
    ].set_index(F.Period.value).to_dict()
    
    return form_years, form_predictors


def get_updated_location_future_data(form: dict) -> pd.DataFrame:
    """Обновляет предполагаемые будущие данные для страны теми, которые пользователь
    определил в форме

    Args:
        form (dict): данные формы из POST запроса в формате:
            имя_поля: значение_поля
            Имя поля составлено в формате:
            {имя_признака_в_таблице}_{год}: значение поля из формы

    Returns:
        pd.DataFrame: _description_
    """
    # Скопируем даныне запроса, чтобы не мутировать их
    form = form.copy()
    # Код страны (убираем это поле из словаря)
    location_code = form.pop('location_code')
    # Предполагаемые будущие данные страны
    location_future_data = get_location_data(future_data, location_code)
    # Словарь, в который будем собирать данные из формы, 
    # чтобы потом присвоить их признакам в таблице
    new_values = {}

    # Итерируемся по всем заполненным полям формы
    for form_field_name, form_field_value in form.items():
        # Разделяем имя поля формы на имя признака таблицы и год
        feature_name, year = form_field_name.split('_')
        # Если данные признака еще не заполняли, зададим пустой список
        if (feature_name not in new_values):
            new_values[feature_name] = []
        # Собираем словарь вида имя_поля_таблицы: список_значений_из_формы
        new_values[feature_name].append(form_field_value)
    
    # Итерируемся по созданному словарю
    for feature_name, values in new_values.items():
        # Меняем в таблице значения на новые (которые были указаны в форме)
        location_future_data[feature_name] = values
        # Приводим столбец к числовому виду
        location_future_data[feature_name] = location_future_data[feature_name].astype(float)
    
    # Возвращаем таблицу с предполагаемыми будущими данными страны,
    # которые обновлены данными, заполненными в форме
    return location_future_data


def get_location_past_data(code: str) -> pd.DataFrame:
    """Возвращает таблицу с данными страны за прошлые года 
        (нужно для построения графика на последнем шаге)

    Args:
        code (str): код страны

    Returns:
        pd.DataFrame: таблица с данными страны за прошлые года
    """
    location_past_data = get_location_data(past_data, code)
    
    return location_past_data