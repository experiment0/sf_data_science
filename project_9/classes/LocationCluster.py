import sys
# Добавим папку проекта в список системных директорий, чтобы Python видел путь к папке utils
sys.path.append('..')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from utils.constants import (
    F, 
    RANDOM_STATE,
)
from utils.prepare_data import (
    move_column_to_end_table,
)


class LocationCluster:
    def __init__(
        self,
        data_source: pd.DataFrame,
        clusters_count: int,
    ):
        # Копируем таблицу, чтобы не мутировать данные
        data = data_source.copy()
        
        # Удалим столбец с кластером на тот случай, если он уже есть в таблице
        if (F.ClusterKMeans.value in list(data.columns)):
            data.drop(columns=[F.ClusterKMeans.value], inplace=True)
        
        self.data = data
        self.clusters_count = clusters_count
        
    
    def __create_mean_scaled_data(self):
        """Создает и сохраняет таблицу со средними масштабированными значениями по каждой стране
        """
        columns_to_drop = [
            F.Period.value, F.ParentLocationCode.value, 
            F.ParentLocation.value, F.Location.value,
        ]

        # Посчитаем средние значения признаков по всем странам
        mean_data = self.data \
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

        self.mean_scaled_data = mean_scaled_data
    
    
    def __calculate_cluster(self):
        """Определяет кластеры для стран и добавляет столбец с их значениями в таблицу
            со средними масштабированными значениями по странам
        """
        # Создадим объект модели
        k_means_model = KMeans(n_clusters=self.clusters_count, random_state=RANDOM_STATE)
        # Обучим модель
        k_means_model.fit_predict(self.mean_scaled_data)
        # Создадим столбец с метками кластеров
        self.mean_scaled_data[F.ClusterKMeans.value] = k_means_model.labels_
        
    
    def create_cluster(self):
        """Реализует алгоритм добавления признака кластера в основную таблицу
        """
        # Создадим и сохраним таблицу со средними масштабированными значениями по каждой стране
        self.__create_mean_scaled_data()
        
        # Определим кластеры для таблицы со средними масштабированными значениями по странам
        self.__calculate_cluster()
        
        # Добавим признак с кластером в исходную таблицу 
        self.mean_scaled_data = self.mean_scaled_data.reset_index() \
            .rename(columns={ 'index': F.SpatialDimValueCode.value })
        
        self.data = self.data.merge(
            self.mean_scaled_data[[F.SpatialDimValueCode.value, F.ClusterKMeans.value]],
            on=F.SpatialDimValueCode.value,
            how='left',
        )
        self.data = move_column_to_end_table(self.data, F.LifeExpectancy.value)