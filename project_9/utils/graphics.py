# Функции для вывода графиков

from IPython.display import display, Markdown
from IPython.core.display import HTML
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import (
    RANDOM_STATE,
    ClusteringAlgorithm,
    TITLE_FONT_SIZE,
    TARGET_FEATURE,
)
from utils.prepare_data import (
    get_temp_rename_dict,
    get_renamed_fields,
)


def dislay_correlation_matrix(
    data: pd.DataFrame, 
    figsize: tuple = (18, 18),
    method: str = 'pearson',
):
    """Выводит матрицу корреляции

    Args:
        data (pd.DataFrame): таблица с данными
        figsize (tuple): размер фигуры. По умолчанию (18, 18)
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
    """
    # Задаем фигуру и координатную плоскость
    fig, axes = plt.subplots(figsize=figsize)
    
    # Параметры тепловой карты
    sns.heatmap(
        data.select_dtypes(include=np.number).corr(method=method), 
        annot=True, 
        vmin=-1, 
        vmax=1, 
        center=0, 
        linewidths=.9, 
        ax=axes, 
        fmt='.3f', 
        cmap='coolwarm'
    )
    plt.title('Тепловая карта корреляции признаков', fontsize=TITLE_FONT_SIZE)
    
    # Выведем график
    plt.show()


def dislay_partial_correlation_matrix(
    data: pd.DataFrame,
    columns_features: list,
    rows_features: list,
    should_rename: bool = True,
    figsize: tuple = (18, 18),
    method: str = 'pearson',
    title: str = 'Тепловая карта корреляции признаков',
    xlabel: str = '',
    ylabel: str = '',
) -> None:
    """Строит матрцу корреляции для выборочных признаков из таблицы

    Args:
        data (pd.DataFrame): таблица с исходными данными
        columns_features (list): признаки, которые должны быть отображены в колонках
        rows_features (list): признаки, которые должны быть отображены в строках
        should_rename (bool, optional): нужно ли переименовать признаки 
            (это может быть удобно, если имена признаков различаются только цифрами). 
            По умолчанию True.
        figsize (tuple, optional): Размер графика. По умолчанию (18, 18).
        method (str, optional): метод рассчета корреляции. По умолчанию 'pearson'.
        title (str, optional): заголовок графика. 
            По умолчанию 'Тепловая карта корреляции признаков'.
        xlabel (str, optional): подпись по оси OX. По умолчанию ''.
        ylabel (str, optional): подпись по оси OY. По умолчанию ''.
    """
    # Оставим в таблице только числовые признаки
    numeric_data = data.select_dtypes(include=np.number)
    
    # Если нужно переименовать признаки
    if (should_rename):
        # Переименуем признаки в таблице
        rename_dict = get_temp_rename_dict(data)
        numeric_data.rename(columns=rename_dict, inplace=True)
    
    # Посчитаем матрицу корреляции
    corr_data = numeric_data.corr(method=method)
    
    # Составим списки признаков для столбцов и строк
    if (should_rename):
        columns_features_result = get_renamed_fields(columns_features)
        rows_features_result = get_renamed_fields(rows_features)
    else:
        columns_features_result = columns_features
        rows_features_result = rows_features
    
    # Оставим в таблице только нужные признаки в столбцах и строках
    corr_data = corr_data[columns_features_result]
    corr_data = corr_data.loc[rows_features_result]
    
    # Задаем фигуру и координатную плоскость
    fig, axes = plt.subplots(figsize=figsize)

    # Параметры тепловой карты
    sns.heatmap(
        corr_data, 
        annot=True, 
        vmin=-1, 
        vmax=1, 
        center=0, 
        linewidths=.9, 
        ax=axes, 
        fmt='.3f', 
        cmap='coolwarm'
    )
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Выведем график
    plt.show()


def display_corellation_by_target(
    data: pd.DataFrame, 
    figsize: tuple = (12, 6),
    method: str = 'pearson', 
    target_feature_name: str = TARGET_FEATURE,
    title: str = None,
    xlabel: str = None, 
    ylabel: str = None,
):
    """Выводит столбчатую диаграмму корреляции признаков с целевой переменной

    Args:
        data (pd.DataFrame): таблица с данными
        figsize (tuple): размер графика. По умолчанию (12, 6)
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
        target_feature_name (str, optional): признак, для которого нужно вывести корреляцию.
            По умолчанию - целевая переменная.
        title (str, optional): заголовок графика. По умолчанию None
        xlabel (str, optional): подпись по оси X. По умолчанию None
        ylabel (str, optional): подпись по оси Y. По умолчанию None
    """
    # Матрица корреляции
    data_corr = data.select_dtypes(include=np.number).corr(method=method)

    # Оставим в таблице только целевую переменную
    data_corr = data_corr[[target_feature_name]]

    # Добавим столбец с типом корреляции
    data_corr['СorrelationType'] = data_corr[target_feature_name]\
        .apply(lambda value: 'positive' if value > 0 else 'negative')
    
    # Цвета столбиков с положительной и отрицательной корреляцией
    positive_color = 'LightCoral'
    negative_color = 'DodgerBlue'
    
    # Добавим столбец с цветом колонок
    data_corr['Color'] = data_corr['СorrelationType']\
        .apply(lambda value: positive_color if value == 'positive' else negative_color)
    
    # Возьмем модуль для всех коэффициентов корреляции
    data_corr[target_feature_name] = data_corr[target_feature_name].apply(abs)

    # Отсортируем данные
    data_corr.sort_values(by=target_feature_name, inplace=True)
    
    # Построим столбчатую диаграмму
    plt.figure(figsize=figsize)
    plt.xticks(rotation=90)

    barplot = sns.barplot(
        data=data_corr, 
        x=data_corr.index, 
        y=target_feature_name,
        palette=list(data_corr['Color'].values)
    )
    
    # Определяем заголовок графика
    if (title is None):
        if (target_feature_name == TARGET_FEATURE):
            title_result = f'Корреляция признаков с целевой переменной {target_feature_name}'
        else:
            title_result = f'Корреляция признаков с переменной {target_feature_name}'
    else:
        title_result = title
    
    barplot.set_title(title_result, fontsize=TITLE_FONT_SIZE)
    
    # Определяем подпись по оси X
    if (xlabel is None):
        xlabel_result = 'Признак'
    else:
        xlabel_result = xlabel
        
    barplot.set_xlabel(xlabel_result)
    
    # Определяем подпись по оси Y
    if (ylabel is None):
        ylabel_result = f'Коэффициент корреляции с {target_feature_name}'
    else:
        ylabel_result = ylabel
        
    barplot.set_ylabel(ylabel_result)
    
    # Задаем сетку
    barplot.grid()
    
    # Выведем график
    plt.show()
    
    # Выведем пояснение к цветам колонок
    legend = '<div style="margin-left: 45px"}>'
    legend += f'<font color="{positive_color}"><strong>Положительная корреляция</strong></font><br />'
    legend += f'<font color="{negative_color}"><strong>Отрицательная корреляция</strong></font>'
    legend += '</div>'
    display(HTML(legend))


def get_inertia(data: pd.DataFrame, clusters_count: int) -> float:
    """Возвращает значение инетрции алгоритма KMeans для переданного числа кластеров

    Args:
        data (pd.DataFrame): данные, которые нужно разделить на кластеры
        clusters_count (int): количество кластеров

    Returns:
        float: значение итерции
    """
    k_means_model = KMeans(n_clusters=clusters_count, random_state=RANDOM_STATE)
    k_means_model.fit(data)
    return k_means_model.inertia_


def display_inertia(data: pd.DataFrame, min_clusters: int, max_clusters) -> None:
    """Выводит график и талицу зависимости инерции от количества кластеров для алгоритма KMeans

    Args:
        data (pd.DataFrame): данные для обучения
        min_clusters (int): минимальное количество кластеров
        max_clusters (_type_): максимальное количество кластеров
    """
    # Cоздадим пустой словарь, ключами будут инерция и количество кластеров
    result_inertia = {'inertia': [], 'clusters_count': []}
    
    for clusters_count in range(min_clusters, max_clusters + 1):
        inertia = get_inertia(data, clusters_count)
        result_inertia['inertia'].append(inertia)
        result_inertia['clusters_count'].append(clusters_count)
    
    result_inertia_data = pd.DataFrame(result_inertia)
    # Визуализируем зависимость значения инерции от количества кластеров
    fig = plt.figure(figsize=(10, 5))
    lineplot = sns.lineplot(data=result_inertia_data, x='clusters_count', y='inertia', marker= 'o')
    lineplot.set_title('Зависимость инерции KMeans от количества кластеров')
    lineplot.set_xlabel('Количество кластеров')
    lineplot.set_ylabel('Значение инерции')
    lineplot.grid()
    plt.show()

    display(result_inertia_data)
    

def get_silhouette(
    data_source: pd.DataFrame, 
    clusters_count: int,
    algoritm: ClusteringAlgorithm = ClusteringAlgorithm.K_MEANS.value,
) -> float:
    """Возвращает значение метрики силуэта для переданного числа кластеров

    Args:
        data_source (pd.DataFrame): данные, которые нужно разделить на кластеры
        clusters_count (int): количество кластеров
        algoritm (ClusteringAlgorithm): название алгоритма для разделения данных.
            По умолчанию 'k_means'.

    Returns:
        float: значение метрики силуэта
    """
    data = data_source.copy()
    
    if (algoritm == ClusteringAlgorithm.K_MEANS.value):
        k_means_model =  KMeans(n_clusters=clusters_count, random_state=RANDOM_STATE)
        k_means_model.fit(data)
        clusters = k_means_model.labels_
    elif (algoritm == ClusteringAlgorithm.EM.value):
        gm_model = GaussianMixture(n_components=clusters_count, random_state=RANDOM_STATE)
        clusters = gm_model.fit_predict(data)
    elif (algoritm == ClusteringAlgorithm.AC.value):
        ac_model = AgglomerativeClustering(n_clusters=clusters_count)
        ac_model.fit(data)
        clusters = ac_model.labels_
        
    # Подсчитаем метрику силуэта, 
    silhouette = silhouette_score(data, clusters, random_state=RANDOM_STATE)
    return silhouette


def display_silhouette(
    data: pd.DataFrame, 
    min_clusters: int, 
    max_clusters: int,
    algoritm: ClusteringAlgorithm = ClusteringAlgorithm.K_MEANS.value,
) -> None:
    """Выводит график и талицу зависимости коэффициента силуэта от количества кластеров

    Args:
        data (pd.DataFrame): данные для обучения
        min_clusters (int): минимальное количество кластеров
        max_clusters (int): максимальное количество кластеров
        algoritm (ClusteringAlgorithm): название алгоритма для разделения данных.
            По умолчанию 'k_means'.
    """
    # Cоздадим пустой словарь, ключами будут значение силуэта и количество кластеров
    result_silhouette = {'silhouette': [], 'clusters_count': []}
    
    for clusters_count in range(min_clusters, max_clusters + 1):
        silhouette = get_silhouette(data, clusters_count, algoritm)
        result_silhouette['silhouette'].append(silhouette)
        result_silhouette['clusters_count'].append(clusters_count)
   
    result_silhouette_data = pd.DataFrame(result_silhouette)    
    # Визуализируем зависимость значения коэффициента силуэта от количества кластеров
    fig = plt.figure(figsize=(10, 5))
    lineplot = sns.lineplot(data=result_silhouette_data, x='clusters_count', y='silhouette', marker= 'o')
    lineplot.set_title('Зависимость коэффициента силуэта от количества кластеров')
    lineplot.set_xlabel('Количество кластеров')
    lineplot.set_ylabel('Значение коэффициента силуэта')
    lineplot.grid()
    plt.show()

    display(result_silhouette_data.sort_values(by='silhouette', ascending=False))


def display_clusters_profiles(
    grouped_data_source: pd.DataFrame, 
    title: str = 'Сравнение средних значений признаков в кластерах стран',
    should_display_table: bool = True,
    is_custom_size: bool = True,
) -> None:
    """Выводит полярную диаграмму профилей кластеров

    Args:
        grouped_data_source (pd.DataFrame): данные профилей 
            (обычно среднее по рассматриваемым признакам профилей)
        title (str): заголовок графика.
            По умолчанию "Распределение средних параметров кластеров"
        should_display_table (bool): нужно ли выводить таблицу со значениями.
            По умолчанию True
        is_custom_size (bool): установить ли свой размер графика
    """
    # Копируем исходные данные, чтобы не мутировать их
    grouped_data = grouped_data_source.copy()
    
    if (should_display_table):
        display(grouped_data)
    
    # Количество кластеров
    clusters_count = len(grouped_data)
    # Создаём список признаков
    features = grouped_data.columns
    
    # Создаём пустую фигуру
    fig = go.Figure()
    
    # В цикле визуализируем полярную диаграмму для каждого кластера
    for i in range(clusters_count):
        # Создаём полярную диаграмму и добавляем её на общий график
        fig.add_trace(go.Scatterpolar(
            r=grouped_data.iloc[i].values, # радиусы
            theta=features, # название засечек
            fill='toself', # заливка многоугольника цветом
            name=f'Cluster {i}', # название — номер кластера
        ))
    
    fig_layout_prams = {
        'title': title, # заголовок графика
        'showlegend': True, # отображение легенды
    }
    if (is_custom_size):
        fig_layout_prams['autosize'] = False # устанавливаем свои размеры графика
        fig_layout_prams['width'] = 800 # ширина (в пикселях)
        fig_layout_prams['height'] = 800 # высота (в пикселях)
        
    # Обновляем параметры фигуры
    fig.update_layout(**fig_layout_prams)
    
    # Отображаем фигуру
    fig.show()
    
    
def display_prediction(
    time_series_train: pd.DataFrame,
    time_series_test: pd.DataFrame,    
    time_series_predict: pd.DataFrame,
    location: str,
    time_series_fitted: pd.DataFrame = None,
) -> None:
    """Выводит график с линиями тренировочных, тестовых и предсказанных значений

    Args:
        time_series_train (pd.DataFrame): временной ряд тренировочной выборки
        time_series_test (pd.DataFrame): временной ряд тестовой выборки
        time_series_predict (pd.DataFrame): временной ряд предсказанной выборки
        location (str): название страны
        time_series_fitted (pd.DataFrame, optional): временной ряд, 
            созданный для тренировочных данных во время обучения модели. 
            По умолчанию None.
    """
    plt.figure(figsize=(10, 4))
    
    (line_train,) = plt.plot(time_series_train, marker="o", color="blue")
    (line_test,) = plt.plot(time_series_test, marker="o", color="aqua")

    if (not time_series_fitted is None):
        (line_fitted,) = plt.plot(time_series_fitted, marker="o", color="pink")
    (line_forecast,) = plt.plot(time_series_predict, marker="o", color="red")

    legend_lines = [line_train, line_test, line_forecast]
    legend_names = ['Тренировочные данные', 'Тестовые данные', 'Прогноз']
    
    if (not time_series_fitted is None):
        legend_lines.append(line_fitted)
        legend_names.append('Ход обучения')
        
    plt.legend(
        legend_lines, 
        legend_names
    )

    plt.title(f'Прогноз ожидаемой продолжительности жизни для страны {location}')
    plt.xlabel('Год')
    plt.ylabel('Продолжительность жизни');
