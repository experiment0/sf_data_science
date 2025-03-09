import sys
# Добавим папку проекта в список системных директорий, чтобы Python видел путь к папке utils
sys.path.append('..')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from utils.constants import (
    F, 
    RANDOM_STATE,
    TARGET_FEATURE,
)
from utils.prepare_data import (
    move_column_to_end_table,
)


class LocationCluster:
    def __init__(
        self,
        clusters_count: int,
        target_feature_name: str = TARGET_FEATURE,
    ):
        """Создает признак кластера стран

        Args:
            clusters_count (int): количество кластеров
            target_feature_name (str, optional): Имя целевой переменной. 
                По умолчанию - константа TARGET_FEATURE.
        """
        self.clusters_count = clusters_count
        self.target_feature_name = target_feature_name
    
    
    def __prepare_data(self, data_source: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает таблицу к добавлению кластера

        Args:
            data_source (pd.DataFrame): исходные данные

        Returns:
            pd.DataFrame: преобразованные данные
        """
        # Копируем таблицу, чтобы не мутировать данные
        data = data_source.copy()
        
        # Удалим столбец с кластером на тот случай, если он уже есть в таблице
        if (F.ClusterKMeans.value in list(data.columns)):
            data.drop(columns=[F.ClusterKMeans.value], inplace=True)
        
        return data
        

    def __get_mean_scaled_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает и возвращает таблицу со средними масштабированными значениями по каждой стране

        Args:
            data (pd.DataFrame): исходная таблица

        Returns:
            pd.DataFrame: таблица со средними масштабированными значениями по каждой стране
        """
        columns_to_drop = [
            F.Period.value, F.ParentLocationCode.value, 
            F.ParentLocation.value, F.Location.value,
        ]

        # Посчитаем средние значения признаков по всем странам
        mean_data = data \
            .drop(columns=columns_to_drop) \
            .groupby(F.SpatialDimValueCode.value) \
            .agg('mean')
            
        # Имена столбцов и названия стран из индекса, которые теряются при преобразовании
        columns = list(mean_data.columns)
        locations_index = list(mean_data.index)

        # Инициализируем нормализатор MinMaxScaler
        mm_scaler = MinMaxScaler()

        # Кодируем исходный датасет
        mean_scaled_data = mm_scaler.fit_transform(mean_data)

        # Добавим имена столбцов и индекс, которые потерялись при преобразовании
        mean_scaled_data = pd.DataFrame(mean_scaled_data, columns=columns, index=locations_index)

        return mean_scaled_data
    

    def __create_cluster(self, mean_scaled_data: pd.DataFrame) -> pd.DataFrame:
        """Определяет кластеры для стран и добавляет столбец с их значениями в таблицу
            со средними масштабированными значениями по странам

        Args:
            mean_scaled_data (pd.DataFrame): таблица со средними масштабированными значениями по странам

        Returns:
            pd.DataFrame: таблица с добавленны в нее столбцом кластера
        """
        # Создадим объект модели
        k_means_model = KMeans(n_clusters=self.clusters_count, random_state=RANDOM_STATE)
        # Обучим модель
        k_means_model.fit_predict(mean_scaled_data)
        # Создадим столбец с метками кластеров
        mean_scaled_data[F.ClusterKMeans.value] = k_means_model.labels_
        
        return mean_scaled_data
    
    
    def fit(self, data_source: pd.DataFrame) -> None:
        """Для входящих данных определят кластер для каждой страны и запоминает эту информацию

        Args:
            data_source (pd.DataFrame): исходные данные
        """
        # Подготовим данные для определения кластера
        data = self.__prepare_data(data_source)
        
        # Создадим таблицу со средними масштабированными значениями по каждой стране
        mean_scaled_data = self.__get_mean_scaled_data(data)
        
        # Определим кластеры стран
        mean_scaled_data = self.__create_cluster(mean_scaled_data)
        
        # Сохраним таблицу с соответствием кода страны и кластера
        location_cluster_data = mean_scaled_data.reset_index() \
            .rename(columns={ 'index': F.SpatialDimValueCode.value })
            
        location_cluster_data = location_cluster_data[[F.SpatialDimValueCode.value, F.ClusterKMeans.value]]
        
        self.location_cluster_data = location_cluster_data
        
    
    def add_cluster(self, data_source: pd.DataFrame) -> pd.DataFrame:
        """Добавляет столбец с кластером страны в таблицу

        Args:
            data_source (pd.DataFrame): исходная таблица

        Returns:
            pd.DataFrame: таблица с добавленным столбцом кластера
        """
        # Подготовим данные для добавления кластера
        data = self.__prepare_data(data_source)
        
        # Добавим кластер
        data = data.merge(
            self.location_cluster_data,
            on=F.SpatialDimValueCode.value,
            how='left',
        )
        
        # Переставим столбец с целевой переменной в конец таблицы
        data = move_column_to_end_table(data, self.target_feature_name)
        
        return data