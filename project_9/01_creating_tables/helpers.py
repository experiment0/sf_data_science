# Содержит вспомогательные функции для объединения таблиц.
# Данные функции используются только в рамках данной папки
# и не вынесени в файлы ../utils/helpers.py или ../utils/prepare_data.py,
# чтобы не загромождать их.

import pandas as pd
import numpy as np


def get_unique_columns(data: pd.DataFrame) -> list:
    """Возвращает столбцы таблицы, которые содержат более одного уникального значения

    Args:
        data (pd.DataFrame): таблица

    Returns:
        list: столбцы, которые содержат более одного уникального значения
    """
    unique_columns = []
    all_columns = list(data.columns)
    
    for column_name in all_columns:
        if (data[column_name].nunique() > 1):
            unique_columns.append(column_name)
    
    return unique_columns


def get_omissions(data: pd.DataFrame) -> pd.DataFrame:
    """Возвращает строки, в которых есть пропуски

    Args:
        data (pd.DataFrame): исходная таблица, в которой нужно найти строки с пропусками

    Returns:
        pd.DataFrame: таблица, содержащая только строки с пропусками
    """
    mask = data.isna().any(axis=1)
    
    return data[mask]


def print_difference_of_sets(
    values_1: np.ndarray, 
    values_2: np.ndarray
) -> None:
    """Выводит информацию о том, по каким элементам различаются два массива

    Args:
        values_1 (np.ndarray): первый массив
        values_2 (np.ndarray): второй массив
    """
    set_1 = set(values_1)
    set_2 = set(values_2)
    
    print(
        'Элементы, которые есть в первом множестве, но отсутствуют во втором: \n',
        set_1 - set_2
    )
    print()
    print(
        'Элементы, которые есть во втором множестве, но отсутствуют в первом: \n',
         set_2 - set_1
    )
    print()


def get_data_with_full_locations_and_years(
    data: pd.DataFrame,
    col_names: dict,
    year_step: int = None,
) -> pd.DataFrame:
    """Добавляет в переданную таблицу строки с пропущенными годами для каждой страны

    Args:
        data (pd.DataFrame): исходная таблица
        col_names (dict): словарь с именами столбцов (код региона, код страны, год)
        year_step (int): шаг, с которым заполняются года. По умолчанию None.

    Returns:
        pd.DataFrame: таблица с добавленными строками пропущенных годов для каждой страны
    """
    # Имя столбца с кодом региона
    col_region_code = col_names['region_code']
    # Имя столбца с кодом страны
    col_location_code = col_names['location_code']
    # Имя столбца с годом
    col_year = col_names['year']
    
    
    def get_region_code(location_code: str) -> str:
        """Возвращает код региона для страны

        Args:
            location_code (str): код страны

        Returns:
            str: код региона, соответствующий переданной стране
        """
        mask = data[col_location_code] == location_code
        region_code = data[mask][col_region_code].unique()[0]
        return region_code
        

    # Получим список с кодами всех стран
    location_codes = list(data[col_location_code].unique())
    # Список со всеми годами
    if (year_step is None):
        years = list(data[col_year].unique())
    else:
        year_min = data[col_year].min()
        year_max = data[col_year].max()
        years = range(year_min, year_max+1, year_step)

    # Создадим вспомогательную таблицу, с заполненными кодами стран и регионов
    locations_data = pd.DataFrame(columns=[col_region_code, col_location_code, col_year])
    locations_data[col_location_code] = location_codes
    locations_data[col_region_code] = locations_data[col_location_code].apply(get_region_code)

    # Создадим пустую таблицу, в которую будем собирать строки с кодами стран для каждого года
    full_locations_data = pd.DataFrame(columns=[col_region_code, col_location_code, col_year])

    # Пройдем по списку всех лет
    for year in years:
        # Во вспомогательной таблице заполним значение года
        locations_data[col_year] = year
        # Добавим таблицу для года в общую
        full_locations_data = pd.concat([
            full_locations_data,
            locations_data,
        ], ignore_index=True)

    # Смержим с таблицей, содержащей строки для всех стран и всех лет 
    # исходную таблицу с данными
    data = full_locations_data.merge(
        data,
        on=[col_region_code, col_location_code, col_year],
        how='left',
    )
    
    return data