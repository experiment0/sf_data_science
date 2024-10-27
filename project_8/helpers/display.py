# Вспомогательные функции для построения графиков и вывода таблиц

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go
from IPython.display import display
from helpers.prepare_data import F


def display_distribution(rfm_data: pd.DataFrame) -> None:
    """Выводит гистрограмму и коробчатую диаграмму для каждого признака в таблице

    Args:
        rfm_data (pd.DataFrame): таблица с данными
    """
    # Список названий признаков
    rfm_features = list(rfm_data.columns)

    # Количество признаков
    rfm_features_count = len(rfm_features)

    # фигура + 2 x rfm_features_count координатных плоскостей
    fig, axes = plt.subplots(2, rfm_features_count, figsize=(18, 9)) 

    for i, feature in enumerate(rfm_features): 
        # гистограмма
        histplot = sns.histplot(data=rfm_data, x=feature, bins=35, ax=axes[0][i])        
        # название графика
        histplot.set_title(f'Гистограмма признака "{feature}"')        
        # поворот отметок на оси абсцисс
        histplot.xaxis.set_tick_params(rotation=45)
        
        # коробчатая диаграмма
        boxplot = sns.boxplot(data=rfm_data[feature].values, orient='h', ax=axes[1][i])        
        # название графика
        boxplot.set_title(f'Коробчатая диаграмма признака "{feature}"')        
        # поворот отметок на оси абсцисс
        boxplot.xaxis.set_tick_params(rotation=45)

    # выравнивание графиков
    plt.tight_layout()


def display_components_distribution(components: np.ndarray, title: str) -> None:
    """Выводит диаграмму распределения главных компонент в 2-х мерном пространстве

    Args:
        components (np.ndarray): массив с 2-мя главными компонентами
        title (str): заголовок графика
    """
    # фигура + координатная плоскость
    fig, ax = plt.subplots(figsize=(18, 9)) 

    # диаграмма рассеяния
    ax.scatter(components[:,0], components[:,1], alpha=0.7) 

    # название графика
    ax.set_title(title)
    # название оси абсцисс
    ax.set_xlabel('Component 1')        
    # название оси ординат
    ax.set_ylabel('Component 2') 


def display_components_clustering(
    components: np.ndarray, 
    labels: np.ndarray, 
    palette: dict,
    title: str,
) -> None:
    """Выводит результат кластеризации данных

    Args:
        components (np.ndarray): данные для кластеризации
        labels (np.ndarray): значения кластеров
        palette (dict): словарь с палитрой для каждого кластера
        title (str): название графика
    """
    # фигура + координатная плоскость
    fig, ax = plt.subplots(figsize = (18, 9)) 

    # диаграмма рассеяния
    sns.scatterplot(
        x=components[:,0], # ось абсцисс
        y=components[:,1], # ось ординат
        hue=labels, # группировка
        palette=palette, # словарь с соответствием значений из labels цвету точки
        ax=ax, # координатная плоскость
    );

    # название графика
    ax.set_title(title)
    # название оси абсцисс
    ax.set_xlabel('Component 1')        
    # название оси ординат
    ax.set_ylabel('Component 2') 


def display_cluster_profiles(data: pd.DataFrame) -> None:
    """Выводит полярную диаграмму для каждого кластера в переданных данных

    Args:
        data (pd.DataFrame): данные профилей кластеров 
            (средние значения характеристик для каждого признака в кластере)
    """
    # Определяем количество кластеров
    clusters_count = data.shape[0]
    
    # Нормализуем данные (приведем к масштабу [0, 1])
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    # Выделяем названия признаков
    features = data.columns
    
    # Создаём пустую фигуру
    fig = go.Figure()
    
    # В цикле визуализируем полярную диаграмму для каждого кластера
    for i in range(clusters_count):
        # Создаём полярную диаграмму и добавляем её на общий график
        fig.add_trace(go.Scatterpolar(
            r=data.iloc[i].values, # радиусы
            theta=features, # название засечек
            fill='toself', # заливка многоугольника цветом
            name=f'Cluster {i}', # название — номер кластера
        ))
        
    # Обновляем параметры фигуры
    fig.update_layout(
        showlegend=True, # отображение легенды
        autosize=False, # устанавливаем свои размеры графика
        width=800, # ширина (в пикселях)
        height=800, # высота (в пикселях)
    )
    
    # Отображаем фигуру
    fig.show()
    

def display_cluster_describe(
    data: pd.DataFrame, 
    cluster: int, 
    cluster_field_name: str = F.LABEL_KMEANS.value
):
    """Выводит характеристики распределения данных для кластера

    Args:
        data (pd.DataFrame): таблица с характеристиками
        cluster (int): номер кластера
        cluster_field_name (str, optional): имя поля, в котором содержатся метки кластеров. 
            По умолчанию F.LABEL_KMEANS.value.
    """
    mask = data[cluster_field_name] == cluster
    print(f'Характеристики распределения данных для кластера {cluster}')
    display(data.drop(columns=[cluster_field_name])[mask].describe())