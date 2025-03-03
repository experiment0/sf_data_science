# Функции для обработки данных

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from utils.constants import (
    TEST_YEARS_COUNT,
    F,
    FieldSemanticCategory,
    fields,
)


# Загружаем таблицу с данными о странах и регионах
regions_and_locations_data = pd.read_csv(
    '../data/regions_and_locations.csv'
)
# Загружаем таблицу с данными о детской смертности
child_mortality_data = pd.read_csv(
    '../data/who_child_0_5_mortality_prepared.csv'
)
# Загружаем таблицу с порядком интегрирования рядов продолжительности жизни разных стран
integration_order_data = pd.read_csv(
    '../data/integration_order.csv'
)


def move_column_to_end_table(
    data_source: pd.DataFrame, 
    column_name: str,
) -> pd.DataFrame:
    """Переставляет столбец с переданным именем в конец таблицы

    Args:
        data_source (pd.DataFrame): исходная таблица
        column_name (str): имя столбца, который нужно переставить в конец

    Returns:
        pd.DataFrame: таблица с переставленным столбцом
    """
    data = data_source.copy()
    columns = list(data.columns)
    columns.append(
        columns.pop(
            columns.index(column_name)
        )
    )
    data = data[columns]
    return data


def get_fields_by_semantic_category(
    semantic_category: FieldSemanticCategory
) -> list:
    """Возвращает список имен полей для переданной смысловой категории

    Args:
        semantic_category (FieldSemanticCategory): смысловая категория полей таблицы

    Returns:
        list: список полей для переданной категории
    """
    fields_list = []
    
    # Пройдем по списку данных полей таблицы
    for field_name, field_data in fields.items():
        if (
            # Если поле относится к переданной категории
            'semantic_category' in field_data and \
            field_data['semantic_category'] == semantic_category
        ):
            # Добавим поле в список
            fields_list.append(field_name)
            
    return fields_list


def add_child_mortality_data(data_source: pd.DataFrame) -> pd.DataFrame:
    """Добавляет в таблицу данные о причинах детской смертности до 5 лет

    Args:
        data_source (pd.DataFrame): исходная таблица

    Returns:
        pd.DataFrame: таблица с добавленными данными
    """
    # Копируем таблицу, чтобы не мутировать исходные данные
    data = data_source.copy()
    # Данные о причинах детской смертности представлены за 2000 - 2017 гг.
    # поэтому оставим в исходной таблице только данные за этот период
    year_max = child_mortality_data[F.Period.value].max()
    mask = data[F.Period.value] <= year_max
    data = data[mask]
    
    # Добавим в таблицу данные о причинах детской смертности до 5 лет
    data = data.merge(
        child_mortality_data,
        on=[F.SpatialDimValueCode.value, F.Period.value],
        how='left',
    )
    
    return data


def get_temp_rename_dict(data: pd.DataFrame) -> dict:
    """Возвращает словарь для переименования столбцов на графике

    Args:
        data (pd.DataFrame): исходная таблица

    Returns:
        dict: словарь для временного переименования столбцов
    """
    # Словарь для переименования, который будем заполнять
    rename_dict = dict()
    
    # Для каждого поля таблицы сделаем проверку
    for field_name in list(data.columns):
        # Если данные для поля есть в основном словаре данных полей 
        # и задано значение для его временного переименования
        if (field_name in fields and 'temp_rename' in fields[field_name]):
            # Запомним значение для переименования
            rename_dict[field_name] = fields[field_name]['temp_rename']
    
    return rename_dict


def get_renamed_fields(field_names: list) -> list:
    """Возвращает список переименованных столбцов

    Args:
        field_names (list): список с исходными именами столбцов

    Returns:
        list: список с переименованными столбцами
    """
    renamed_fields = []
    
    for field_name in field_names:
        if (field_name in fields and 'temp_rename' in fields[field_name]):
            renamed_fields.append(fields[field_name]['temp_rename'])
            
    return renamed_fields


def get_numerical_data_and_object_columns(data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Убирает стобцы со строковыми значениями из таблицы. 
    Возвращает обновленную таблицу и список имен полей со строковыми значениями.

    Args:
        data (pd.DataFrame): исходная таблица

    Returns:
        Tuple[pd.DataFrame, list]: измененная таблица 
            и список имен полей со строковыми значениями.
    """
    object_columns = data.select_dtypes(include='object').columns.to_list()
    numerical_data = data.drop(columns=object_columns)
    
    return numerical_data, object_columns


def get_predictors(data: pd.DataFrame) -> list:
    """Возвращает список столбцов, которые являются предикторами

    Args:
        data (pd.DataFrame): исходная таблица

    Returns:
        list: список столбцов-предикторов
    """
    fields_list = []
    data_columns = list(data.columns)
    
    # Пройдем по списку данных полей таблицы
    for field_name, field_data in fields.items():
        if (field_data['is_predictor'] and field_name in data_columns):
            fields_list.append(field_name)
            
    return fields_list


def get_train_test_data(
    data: pd.DataFrame, 
    test_years_count: int = TEST_YEARS_COUNT, 
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет данные таблицы на 2 части (тренировочные и тестовые)

    Args:
        data (pd.DataFrame): Исходные данные
        test_years_count (int, optional): Количество последних лет, которые отделяем в тестовую выборку. 
            По умолчанию TEST_YEARS_COUNT.

    Raises:
        ValueError: Проверка, что количество лет для тестовой выборки не больше, чем рассматриваемый период.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            train_data - тренировочные данные, 
            test_data - тестовые данные
    """
    # Минимальный и максимальный года, за которые рассматриваем данные
    year_min = data[F.Period.value].min()
    year_max = data[F.Period.value].max()
    # Год, по которому разделяем выборки на тренировочную и тестовую
    boundary_year = year_max - test_years_count
    # Проверяем корректность входящих параметров
    if (boundary_year <= year_min):
        raise ValueError(f'Количество лет для предсказания должно быть меньше {year_max - year_min - 1}.')
    
    # Разделим данные на тренировочные и тестовые
    train_data = data[data[F.Period.value] <= boundary_year]
    test_data = data[data[F.Period.value] > boundary_year]
    
    return train_data, test_data


def extract_predictors(
    data: pd.DataFrame,
    should_include_time_feature: bool = True,
    additional_predictors: list = [],
) -> pd.DataFrame:
    """Извлекает предикторы из таблицы

    Args:
        data (pd.DataFrame): Исходные данные
        should_include_time_feature (bool, optional): Включать ли в тренировочные данные столбец с годом. 
            По умолчанию True.
        additional_predictors (list, optional): Список дополнительных предикторов.
            По умолчанию []

    Returns:
        pd.DataFrame: Таблица, состоящая только из предикторов
    """
    # Формируем список с предикторами
    predictors = get_predictors(data) + additional_predictors
    if (not should_include_time_feature):
        predictors.remove(F.Period.value)
    
    # Список колонок исходноу таблицы
    columns = list(data.columns)
    
    # Оставим колонки, которые есть и в таблице и в предикторах
    result_predictors = list(set(predictors) & set(columns))

    return data[result_predictors]


def get_train_test_split(
    data: pd.DataFrame, 
    test_years_count: int = TEST_YEARS_COUNT, 
    should_include_time_feature: bool = True,
    additional_predictors: list = [],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Формирует и возвращает данные для тренировочной и тестовой выбоки

    Args:
        data (pd.DataFrame): Исходные данные
        test_years_count (int, optional): Количество последних лет, которые отделяем в тестовую выборку. 
            По умолчанию TEST_YEARS_COUNT.
        should_include_time_feature (bool, optional): Включать ли в тренировочные данные столбец с годом. 
            По умолчанию True.
        additional_predictors (list, optional): Список дополнительных предикторов.
            По умолчанию []

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train - предикторы тренировочной выборки, 
            X_test - предикторы тестовой выборки, 
            y_train - целевая переменная тренировочной выборки, 
            y_test - целевая переменная тестовой выборки
    """
    # Разделим данные на тренировочные и тестовые    
    train_data, test_data = get_train_test_data(data, test_years_count)
    
    # Формируем список с предикторами
    predictors = get_predictors(data) + additional_predictors
    if (not should_include_time_feature):
        predictors.remove(F.Period.value)
    
    # Разделяем данные на предикторы и целевую переменную
    X_train = extract_predictors(train_data, should_include_time_feature, additional_predictors)
    X_test = extract_predictors(test_data, should_include_time_feature, additional_predictors)
    y_train = train_data[F.LifeExpectancy.value]
    y_test = test_data[F.LifeExpectancy.value]
    
    return X_train, X_test, y_train, y_test


def get_location_codes(data: pd.DataFrame) -> list:
    """Возвращает список из кодов стран, встречающихся в таблице

    Args:
        data (pd.DataFrame): исходная таблица с данными

    Returns:
        list: список с кодами стран
    """
    location_codes = list(data[F.SpatialDimValueCode.value].unique())
    return location_codes


def get_location_data(data: pd.DataFrame, code: str) -> pd.DataFrame:
    """Возвращает данные для страны с переданным кодом

    Args:
        data (pd.DataFrame): исходные данные
        code (str): код страны

    Returns:
        pd.DataFrame: отфильтрованные данные для страны
    """
    mask = data[F.SpatialDimValueCode.value] == code
    return data[mask]


def get_location_by_code(code: str) -> str:
    """Возвращает название страны по ее коду

    Args:
        code (str): код страны

    Returns:
        str: название страны
    """
    mask = regions_and_locations_data[F.SpatialDimValueCode.value] == code
    location = regions_and_locations_data[mask][F.Location.value]
    
    return location.values[0]


def convert_year_to_datetime(data_source: pd.DataFrame) -> pd.DataFrame:
    """Переводит столбец с годом в столбец с типом datetime

    Args:
        data_source (pd.DataFrame): исходная таблица

    Returns:
        pd.DataFrame: таблица, в которой признак года переведен в столбец с типом datetime
    """
    data = data_source.copy()
    
    # Переведем год в признак с типом datetime
    data[F.Period.value] = data[F.Period.value].astype(str)
    data[F.Period.value] = pd.to_datetime(
        data[F.Period.value], 
        # %Y-%m-%d
        format='%Y'
    )
    
    # Значения ожидаемой продолжительности жизни при рождении от ООН 
    # соответствуют оценкам на середину года, согласующимся 
    # с соответствующими пятилетними прогнозами населения ООН по варианту рождаемости.
    # https://www.who.int/data/gho/data/indicators/indicator-details/GHO/life-expectancy-at-birth-(years)
    # Не будем прибавлять полгода, так как это не удобно для чтения графиков.
    # data[F.Period.value] = data[F.Period.value] + pd.DateOffset(months=6)
    
    return data
    
    
def convert_year_to_date_index(data_source: pd.DataFrame) -> pd.DataFrame:
    """Переводит столбец с годом в индекс с типом datetime

    Args:
        data_source (pd.DataFrame): исходная таблица

    Returns:
        pd.DataFrame: таблица, в которой признак года переведен в индекс с типом datetime
    """
    data = data_source.copy()
    
    # Переведем год в признак с типом datetime
    data = convert_year_to_datetime(data)
    
    # Переведем признак в индекс
    data.set_index(F.Period.value, inplace=True)
    
    return data


def get_location_time_series(
    data: pd.DataFrame, 
    code: str,
    value_feature_name: str = F.LifeExpectancy.value,
) -> pd.Series:
    """Возвращает данные с временным рядом для заданной страны

    Args:
        data (pd.DataFrame): исходные данные
        code (str): код страны
        value_feature_name: (str, optional): имя признака, 
            из значений которого составляется временной ряд.
            По умолчанию - целевой признак продолжительности жизни LifeExpectancy

    Returns:
        pd.Series: сформированный временной ряд
    """
    # Маска для страны
    location_mask = data[F.SpatialDimValueCode.value] == code
    # Отделяем данные для заданной страны
    location_data = data[location_mask]
    # Оставляем только временной и целевой признаки
    location_time_series = location_data[[F.Period.value, value_feature_name]]
    # Переводим столбец с годом в индекс с типом datetime
    location_time_series = convert_year_to_date_index(location_time_series)
    
    return location_time_series


def get_formatted_time_series(values: np.ndarray, year_start: int) -> pd.DataFrame:
    """Формирует таблицу с временным рядом

    Args:
        values (np.ndarray): значения продолжительности жизни
        year_start (int): год, с которого начинается временной ряд

    Returns:
        pd.DataFrame: таблица с временным рядом
    """
    # Количество лет временного ряда
    years_count = len(values)
    # Список лет временного ряда
    years_list = list(range(year_start, year_start + years_count))
    # Сформируем таблицу с данными
    time_series = pd.DataFrame({
        F.Period.value: years_list,
        F.LifeExpectancy.value: values,
    })
    # Переводим столбец с годом в индекс с типом datetime
    time_series = convert_year_to_date_index(time_series)
    
    return time_series


def get_data_with_smoothing_target_feature(
    train_data_source: pd.DataFrame,
    test_data_source: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Добавляет признак со сглаженными значениями продолжительности жизни.
       Тренировочные данные формируются в ходе обучения модели. Тестовые определяются путем прогноза. 

    Args:
        train_data_source (pd.DataFrame): тренировочные данные
        test_data_source (pd.DataFrame): тестовые данные (или данные для предсказания)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            train_data - тренировочные данные с добавленным признаком, 
            test_data - тестовые данные с добавленным признаком
    """
    # Копируем таблицы, чтобы не мутировать их
    train_data = train_data_source.copy()
    test_data = test_data_source.copy()
    
    # На всякий случай сортируем таблицы, хотя они должны приходить отсортированные
    train_data.sort_values(by=[F.SpatialDimValueCode.value, F.Period.value], inplace=True)
    test_data.sort_values(by=[F.SpatialDimValueCode.value, F.Period.value], inplace=True)
    
    # Коды всех стран
    location_codes = get_location_codes(train_data)

    # Список с годами тренировочной таблицы
    years_train_list = sorted(list(train_data[F.Period.value].unique()))
    # Список с годами тестовой таблицы
    years_test_list = sorted(list(test_data[F.Period.value].unique()))
    # Количество предсказаний, которое будем делать
    predictions_count = len(years_test_list)

    # Список столбцов для таблицы со сглаженными значениями
    smoothing_columns = [F.SpatialDimValueCode.value, F.Period.value, F.SmoothingLifeExpectancy.value]
    # Таблица, в которую будем собирать тренировочные сглаженные значения 
    smoothing_train_data = pd.DataFrame(columns=smoothing_columns)
    # Таблица, в которую будем собирать значения прогноза
    smoothing_test_data = pd.DataFrame(columns=smoothing_columns)
    
    # Для каждой страны посчитаем сглаженные значения
    for code in location_codes:
        # Временной ряд для тренировочных данных
        location_time_series_train = get_location_time_series(train_data, code)
        
        # Получим объект с моделью
        exp_smoothing_model = ExponentialSmoothing(
            location_time_series_train, # тренировочные данные
            trend='add', # тип тренда - аддитивный
            damped_trend=True, # затухание тренда
        )
        # Обучим модель
        exp_smoothing_model_fit = exp_smoothing_model.fit(
            smoothing_level=0.85, # альфа - коэффициент сглаживания для уровня
            smoothing_trend=0.5, # beta - коэффициент сглаживания тренда
        )
        # Получим прогноз на то же количество лет, которое содержится в тестовом временном ряду
        forecast = exp_smoothing_model_fit.forecast(predictions_count)
        
        # Сформируем таблицу с тренировочными сглаженными значениями
        location_smoothing_train_data = pd.DataFrame({
            F.SpatialDimValueCode.value: code,
            F.Period.value: years_train_list,
            F.SmoothingLifeExpectancy.value: exp_smoothing_model_fit.fittedvalues.values,
        })
        # Сформируем таблицу с полученным прогнозом
        location_smoothing_test_data = pd.DataFrame({
            F.SpatialDimValueCode.value: code,
            F.Period.value: years_test_list,
            F.SmoothingLifeExpectancy.value: forecast.values,
        })
        
        # Добавим тренировочные сглаженные значения в общую таблицу 
        smoothing_train_data = pd.concat(
            [smoothing_train_data, location_smoothing_train_data], 
            ignore_index=True
        )
        # Добавим значения прогноза в общую таблицу
        smoothing_test_data = pd.concat(
            [smoothing_test_data, location_smoothing_test_data],
            ignore_index=True
        )
   
    # Добавим столбец с сглаженными тренировочными значениями в исходную тренировочную таблицу
    train_data = train_data.merge(
        smoothing_train_data,
        on=[F.SpatialDimValueCode.value, F.Period.value],
        how='left',
    )
    # Добавим столбец с прогнозами в тестовую таблицу
    test_data = test_data.merge(
        smoothing_test_data,
        on=[F.SpatialDimValueCode.value, F.Period.value],
        how='left',
    )
    
    train_data[F.Period.value] = train_data[F.Period.value].astype(int)
    test_data[F.Period.value] = test_data[F.Period.value].astype(int)
    
    # Переведем новый столбец в тип float
    train_data[F.SmoothingLifeExpectancy.value] = \
      train_data[F.SmoothingLifeExpectancy.value].astype(float)
    test_data[F.SmoothingLifeExpectancy.value] = \
      test_data[F.SmoothingLifeExpectancy.value].astype(float)
    
    return train_data, test_data


def get_scaled_data(
    data_source: pd.DataFrame, 
    cluster_feature: str = F.ClusterKMeans.value,
    should_exclude_time_feature: bool = True,
) -> pd.DataFrame:
    """Возвращает масштабированные данные

    Args:
        data_source (pd.DataFrame): исходные данные
        cluster_feature (str, optional): имя признака с кластером. 
            По умолчанию F.ClusterKMeans.value.
        should_exclude_time_feature (bool, optional): нужно ли исключить признак с годом. 
            По умолчанию True.

    Returns:
        pd.DataFrame: преобразованные данные
    """
    # Скопируем данные, чтобы не мутировать исходную таблицу
    data = data_source.copy()
    # Список колонок
    columns = list(data.columns)
    
    # Удалим столбец с кластером (масштабировать признак кластера нет смысла)
    if (cluster_feature in columns):
        data.drop(columns=[cluster_feature], inplace=True)
    
    # Если необходимо удалить признак года, тоже его удалим, 
    # поскольку масштабировать год тоже нет смысла
    if (should_exclude_time_feature and F.Period.value in columns):
        data.drop(columns=[F.Period.value], inplace=True)
    
    # Оставим только числовые столбцы
    numeric_data = data.select_dtypes(include=np.number)

    # Создаем объект класса MinMaxScaler для масштабирования
    scaler = MinMaxScaler()
    
    # Масштабируем данные
    scaled_data = pd.DataFrame(
        scaler.fit_transform(numeric_data), 
        columns=numeric_data.columns,
        index=numeric_data.index,
    )
    
    # Вернем столбец с кластером
    if (cluster_feature in columns):
        scaled_data[cluster_feature] = data_source[cluster_feature]
    
    return scaled_data


def get_integration_order(code: str) -> int:
    """Возвращает порядок интегрирования ряда ожидаемой продолжительности жизни
    для страны, код которой передан

    Args:
        code (str): код страны

    Returns:
        int: порядок интегрирования
    """
    mask = integration_order_data[F.SpatialDimValueCode.value] == code
    integration_order = integration_order_data[mask]['IntegrationOrder'].values[0]
    
    return integration_order