import sys
# Добавим папку проекта в список системных директорий, чтобы Python видел путь к папке utils
sys.path.append('..')

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.constants import (
    F, 
    TARGET_FEATURE,
)
from utils.prepare_data import (
    move_column_to_end_table,
    move_column_to_end_table,
)
from utils.helpers import get_cosine


class LocationCoef:
    def __init__(
        self,
        columns_to_exclude: list,
        method: str = 'pearson',
        target_feature_name: str = TARGET_FEATURE,
    ):
        """Добавляет в таблицу признаки с коэффициентами, которые отражают
            благополучие и неблагополучие стран.
            Подробнее принцип формирования коэффициентов описан здесь ../04_feature_inginiring/03_coef.ipynb

        Args:
            columns_to_exclude (list): Список столбцов, которые не будут участвовать в формировании признаков
            method (str, optional): Метод, с помощью которого считается коэффициент корреляции. 
                По умолчанию 'pearson'.
            target_feature_name (str, optional): Имя целевой переменной. 
                По умолчанию - константа TARGET_FEATURE.
        """
        self.columns_to_exclude = columns_to_exclude
        self.method = method
        self.target_feature_name = target_feature_name
    
    
    def __prepared_data(self, data_source: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает исходные данные к добавлению признаков с коэффициентами

        Args:
            data_source (pd.DataFrame): исходные данные

        Returns:
            pd.DataFrame: подготовленные данные
        """
        # Копируем таблицу, чтобы не мутировать данные
        data = data_source.copy()
        
        # Для случая, если запускаем класс для таблицы с добавленными коэффициентами, удалим поля с ними
        columns = list(data.columns)

        if (F.PositiveCoef.value in columns):
            data.drop(columns=[F.PositiveCoef.value], inplace=True)
        if (F.NegativeCoef.value in columns):
            data.drop(columns=[F.NegativeCoef.value], inplace=True)
        
        return data
    
    
    def __get_split_correlation_fields(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Формирует две таблицы. 
        Одну с признаками, положительно коррелирующими с целевой переменной 
        (в формате: индекс - имя признака, значение - коэффициент корреляции).
        И вторую аналогичную с отрицательно коррелирующими признаками
        
        Args:
            data (pd.DataFrame): исходная таблица, по данным которой смотрим корреляцию
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: таблицы данными признаков корреляции
        """
        # Уберем столбцы, которые не будут участвовать в формировании коэффициентов
        data = data.drop(columns=self.columns_to_exclude)
        
        # Матрица корреляции
        data_corr = data.select_dtypes(include=np.number).corr(method=self.method)

        # Оставим в таблице только целевую переменную
        data_corr = data_corr[[self.target_feature_name]]
        
        # Имя поля для значений коэффициента корреляции
        corr_field_name = 'Correlation'
        # Переименуем
        data_corr.rename(columns={ self.target_feature_name: corr_field_name }, inplace=True)
        
        # Маски для положительной и отрицательной корреляции
        mask_positive = data_corr[corr_field_name] >= 0
        mask_negative = data_corr[corr_field_name] < 0

        # Строки с положительной корреляцией
        positive_data = data_corr[mask_positive]
        positive_data.sort_values(by=corr_field_name, ascending=False, inplace=True)
        
        # Строки с отрицательной корреляцией
        negative_data = data_corr[mask_negative]
        negative_data.sort_values(by=corr_field_name, inplace=True)    
        
        return positive_data, negative_data
    

    def __get_mean_scaled_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает и возвращает таблицу со средними масштабированными значениями по каждой стране

        Args:
            data (pd.DataFrame): исходная таблица

        Returns:
            pd.DataFrame: таблица со средними масштабированными значениями по каждой стране
        """
        object_columns = [
            F.ParentLocationCode.value,
            F.ParentLocation.value,
            F.Location.value,
        ]
        # Сформируем таблицу со средними значениями по каждой стране
        # Исключим признак с кластером, так как среднее по нему не имеет смысла
        mean_data = data \
            .drop(columns=self.columns_to_exclude + object_columns + [F.ClusterKMeans.value]) \
            .groupby(F.SpatialDimValueCode.value) \
            .agg('mean') 
            
        # Запомним индексы и названия колонок перед масштабированием данных
        index = list(mean_data.index)
        columns = list(mean_data.columns)

        # Инициализируем нормализатор MinMaxScaler
        mm_scaler = MinMaxScaler()

        # Кодируем исходный датасет
        mean_scaled_data = mm_scaler.fit_transform(mean_data)

        # Добавим имена столбцов и индекс, которые потерялись при преобразовании
        mean_scaled_data = pd.DataFrame(mean_scaled_data, columns=columns, index=index)

        # Переименуем индексный столбец с кодами стран
        mean_scaled_data = mean_scaled_data \
            .reset_index().rename(columns={'index': F.SpatialDimValueCode.value})
        
        # Сформируем таблицу с признаками кода региона и кластера
        # Чтобы добавить столбец с кластером к mean_scaled_data
        clusters = data.groupby(F.SpatialDimValueCode.value).agg(
            ClusterKMeans=(F.ClusterKMeans.value, 'first')
        )
        clusters = clusters.reset_index().rename(columns={'index': F.SpatialDimValueCode.value})
        
        # Добавим столбец с кластером
        mean_scaled_data = mean_scaled_data.merge(
            clusters,
            on=F.SpatialDimValueCode.value,
            how='inner',
        )
                    
        return mean_scaled_data
    
    
    def __create_reference_vectors(self, mean_scaled_data: pd.DataFrame) -> None:
        """Составляет векторы признаков условно идеальной страны и максимально неблагополучной страны.

        Args:
            mean_scaled_data (pd.DataFrame): таблица со средними масштабированными значениями по каждой стране
        """
        # Вектор максимально благополучной страны
        positive_vector = dict()
        # Вектор максимально неблагополучной страны
        negative_vector = dict()

        # Страны из благополучного кластера
        good_cluster_data = mean_scaled_data[mean_scaled_data[F.ClusterKMeans.value] == 0]
        # Страны из неблагополучного кластера
        bad_cluster_data = mean_scaled_data[mean_scaled_data[F.ClusterKMeans.value] == 1]

        # Заполним данные вектора максимально благополучной страны
        for field in self.positive_fields:
            max_value = good_cluster_data[field].max()
            positive_vector[field] = max_value

        # Заполним данные вектора максимально неблагополучной страны
        for field in self.negative_fields:
            max_value = bad_cluster_data[field].max()
            negative_vector[field] = max_value

        # Сохраним полученные векторы
        self.positive_vector = positive_vector
        self.negative_vector = negative_vector
    
    
    def __get_positive_coef(self, row: pd.Series) -> float:
        """Подсчитывает косинус угла между вектором данных переданной страны и вектором идеальной страны

        Args:
            row (pd.Series): средние данные по стране 
                (строка таблицы со средними показателями по странам)

        Returns:
            float: косинус угла между вектором данных страны и вектором идеальной страны
        """
        # Переведем данные вектора идеальной страны в массив
        positive_values = np.array(list(self.positive_vector.values()))
        
        # Данные вектора страны, которые будем отбирать из переданной строки
        row_values = []
        
        for field in self.positive_vector.keys():
            row_values.append(row[field])
            
        return get_cosine(positive_values, np.array(row_values))
        

    def __get_negative_coef(self, row: pd.Series) -> float:
        """Подсчитывает косинус угла между вектором данных переданной страны 
            и вектором максимально неблагополучной страны

        Args:
            row (pd.Series): средние данные по стране 
                (строка таблицы со средними показателями по странам)

        Returns:
            float: косинус угла между вектором данных страны и максимально неблагополучной страны
        """
        # Переведем данные вектора максимально неблагополучной страны в массив
        negative_values = np.array(list(self.negative_vector.values()))
    
        # Данные вектора страны, которые будем отбирать из переданной строки
        row_values = []
        
        for field in self.negative_vector.keys():
            row_values.append(row[field])
            
        return get_cosine(negative_values, np.array(row_values))
    
    
    def fit(self, data_source: pd.DataFrame) -> None:
        """Для исходных данных создает и запоминает таблицу с соответствием
            кода страны и ее коэффициентов

        Args:
            data_source (pd.DataFrame): исходная таблица
        """
        # Подготовим данные таблицы для дальнейших манипуляций
        data = self.__prepared_data(data_source)
        
        # Разделим данные по признаку знака коэффициента корреляции с целевой переменной
        positive_data, negative_data = self.__get_split_correlation_fields(data)
        
        # Составим список с именами положительных признаков и сохраним его
        positive_fields = list(positive_data.index)
        self.positive_fields = positive_fields

        # Составим список с именами отрицательных признаков и сохраним его
        negative_fields = list(negative_data.index)
        self.negative_fields = negative_fields
        
        # Создадим таблицу со средними масштабированными значениями по каждой стране
        mean_scaled_data = self.__get_mean_scaled_data(data)
        
        # Cоставим векторы признаков условно идеальной страны и максимально неблагополучной страны
        self.__create_reference_vectors(mean_scaled_data)
        
        # Посчитаем для каждой страны значение коэффициента благополучия
        mean_scaled_data[F.PositiveCoef.value] = \
            mean_scaled_data.apply(lambda row: self.__get_positive_coef(row), axis=1)
        
        # Посчитаем для каждой страны значение коэффициент неблагополучия
        mean_scaled_data[F.NegativeCoef.value] = \
            mean_scaled_data.apply(lambda row: self.__get_negative_coef(row), axis=1)
        
        # Составим таблицу с соответствием кода страны и ее коэффициентов и сохраним
        location_coefs_data = mean_scaled_data[[ 
            F.SpatialDimValueCode.value, F.PositiveCoef.value, F.NegativeCoef.value,
        ]]
        self.location_coefs_data = location_coefs_data
    
    
    def transform(self, data_source: pd.DataFrame) -> pd.DataFrame:
        """Добавляет в переданную таблицу признаки с ранее рассчитанными 
            коэффициентами стран.

        Args:
            data_source (pd.DataFrame): исходная таблица

        Returns:
            pd.DataFrame: таблица с добавленными признаками коэффициентов
        """
        # Подготовим данные таблицы для дальнейших манипуляций
        data = self.__prepared_data(data_source)
        
        # Добавим коэффициенты в таблицу
        data = data.merge(
            self.location_coefs_data,
            on=F.SpatialDimValueCode.value,
            how='left',
        )
        
        # Переставим столбец с таргетом в конец таблицы
        data = move_column_to_end_table(data, self.target_feature_name)
        
        return data
