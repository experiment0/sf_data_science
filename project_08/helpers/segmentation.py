# Вспомогательные функции для сегментации данных

from types import FunctionType
import numpy as np
import pandas as pd
import seaborn as sns 
from IPython.display import display
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


from helpers.constants import (
    RANDOM_STATE,
)


def get_silhouette_kmeans(clusters_count: int, data: np.ndarray) -> float:
    """Возвращает значение коэффициента силуэта для алгоритма KMeans

    Args:
        clusters_count (int): количество кластеров
        data (np.ndarray): данные для кластеризации

    Returns:
        float: значение коэффициента силуэта
    """
    kmeans_model = KMeans(n_clusters=clusters_count, random_state=RANDOM_STATE)
    kmeans_model.fit(data)
    
    silhouette = silhouette_score(data, kmeans_model.labels_)
    
    return silhouette


def get_silhouette_gm(clusters_count: int, data: np.ndarray) -> float:
    """Возвращает значение коэффициента силуэта для EM-алгоритма

    Args:
        clusters_count (int): количество кластеров
        data (np.ndarray): данные для кластеризации

    Returns:
        float: значение коэффициента силуэта
    """
    gm_model = GaussianMixture(n_components=clusters_count, random_state=RANDOM_STATE)
    labels = gm_model.fit_predict(data)
    
    silhouette = silhouette_score(data, labels)
    
    return silhouette


def get_silhouette_ac(clusters_count: int, data: np.ndarray) -> float:
    """Возвращает значение коэффициента силуэта для агломеративной кластеризации

    Args:
        clusters_count (int): количество кластеров
        data (np.ndarray): данные для кластеризации

    Returns:
        float: значение коэффициента силуэта
    """
    ac_model = AgglomerativeClustering(n_clusters=clusters_count)
    ac_model.fit(data)
    
    silhouette = silhouette_score(data, ac_model.labels_)
    
    return silhouette


def get_silhouette_data(
    components: np.ndarray,
    silhouette_function: FunctionType, 
    min_clusters: int, 
    max_clusters: int, 
) -> pd.DataFrame:
    """Возвращает таблицу с данными количества кластеров и соответствующим коэффициентом силуэта

    Args:
        components (np.ndarray): массив с данными для кластеризации
        silhouette_function (FunctionType): функция, с помощью которой находим коэффициент силуэа
        min_clusters (int): минимальное количество кластеров
        max_clusters (int): максимальное количество кластеров

    Returns:
        pd.DataFrame: таблица с данными количества кластеров и соответствующим коэффициентом силуэта
    """
    result = { 'silhouette': [], 'clusters_count': [] }
    
    for clusters_count in range(min_clusters, max_clusters + 1):
        silhouette = silhouette_function(clusters_count, components)
        result['silhouette'].append(silhouette)
        result['clusters_count'].append(clusters_count)
    
    return pd.DataFrame(result)


def display_silhouette_data(
    components: np.ndarray,
    silhouette_function: FunctionType, 
    min_clusters: int, 
    max_clusters: int, 
) -> None:
    """Выводит таблицу с количеством кластеров и соответствующим коэффициентом силуэта,
    а также график изменения коэффициента силуэта в зависимости от количества кластеров

    Args:
        components (np.ndarray): массив с данными для кластеризации
        silhouette_function (FunctionType): функция, с помощью которой находим коэффициент силуэа
        min_clusters (int): минимальное количество кластеров
        max_clusters (int): максимальное количество кластеров
    """
    silhouette_data = get_silhouette_data(components, silhouette_function, min_clusters, max_clusters)
    
    silhouette_data = silhouette_data.sort_values(by='silhouette', ascending=False)
    display(silhouette_data)
    
    lineplot = sns.lineplot(data=silhouette_data, x='clusters_count', y='silhouette', marker='o')
    lineplot.set_title('Значение коэффициента силуэта в зависимости от количества кластеров')
    lineplot.set_xlabel('Количество кластеров')
    lineplot.set_ylabel('Коэффициент силуэта')
    
    first_row = silhouette_data.iloc[0]
    print('Наилучшее разделение будет при количестве кластеров: ', first_row['clusters_count'])
    print('И значении коэффициента силуэта: ', round(first_row['silhouette'], 3))
