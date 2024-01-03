# Содержит функции для вывода информации по исследованию таблиц и признаков

from IPython.display import display, Markdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
import optuna
from pprint import pprint

from functions.constants import TARGET_FEATURE, FieldType, fields_params
from functions.helpers import (
    show_translate_feature_values, 
    get_multicolinear_pairs, 
    get_title,
)
from functions.graphics import (
    show_pie, 
    show_countplot, 
    show_distribution, 
    show_countplot_by_target,
    show_distribution_by_target,
    show_correlation_matrix,
    show_corellation_by_target,
    show_scatterplot,
    show_histplot_by_target,
)


def display_field_description(field_name: str):
    """Выводит описание содержимого столбца таблицы

    Args:
        field_name (str): имя столбца таблицы
    """
    if (field_name in fields_params):
        display(Markdown(f"- `{field_name}` - {fields_params[field_name]['description']}"))
        

def display_field_descriptions(field_names: list[str]):
    """Выводит описания содержимого полей из списка

    Args:
        field_names (list[str]): список и именами полей
    """
    for field_name in field_names:
        display_field_description(field_name)
    
        

def display_fields_info(data_source: pd.DataFrame):
    """Выводит информацию о столбцах таблицы

    Args:
        data_source (pd.DataFrame): таблица с данными
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    display(Markdown('**Выведем первые строки таблицы**'))
    display(data.head())
    
    display(Markdown('**Размер таблицы**'))
    display(data.shape)
    
    display(Markdown('**Информация о столбцах**'))
    display(data.info())
    
    display(Markdown('**Количество пропусков в столбцах**'))
    display(data.isna().sum())
    

def display_fields_distribution(data_source: pd.DataFrame):
    """Выводит информацию о распределении данных в столбцах таблицы

    Args:
        data_source (pd.DataFrame): таблица с данными
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    # Список с именами столбцов таблицы
    fields = data.columns.to_list()
    
    if (len(fields) > 1):
        display(Markdown('Посмотрим на распределение признаков в таблице.'))
    
    for field_name, params in fields_params.items():
        # Если поля нет в списке описаний, переходим к следующему
        if (field_name not in fields):
            continue
        
        display(Markdown(f"### Признак `{field_name}` ({params['description']})"))
        display(Markdown(f'Посмотрим на распределение признака `{field_name}`'))
        
        # Вынимаем параметры для графика
        # Копируем, чтобы не мутировать исходный словарь
        graph_params = params['distribution_graph_params'].copy()
        
        # Заголовок графика
        graph_title = graph_params.pop('title')
        
        # Если признак числовой
        if (params['field_type'] == FieldType.NUMERIC.value):
            # Выводим гистограмму и коробчатые диаграму
            show_distribution(
                data, 
                field_name, 
                graph_title,
                get_title(params['description']),
                **graph_params,
            )
            # Если признак категориальный
        elif (params['field_type'] == FieldType.CATEGORY.value):
            # Выводим столбчатую диаграмму
            show_countplot(
                data, 
                field_name, 
                graph_title,
                get_title(params['description']),
                **graph_params,
            )
            # Выводим перевод значений категориального признака
            show_translate_feature_values(field_name)
            # Если признак бинарный
        elif (params['field_type'] == FieldType.BINARY.value):
            # Выводим круговую диаграмму
            show_pie(
                data,
                field_name,
                graph_title,
            )


def display_fields_distribution_by_target(data_source: pd.DataFrame):
    """Выводит информацию о распределении данных в столбцах таблицы
    в разрезе целевой переменной

    Args:
        data_source (pd.DataFrame): таблица с данными
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    # Список с именами столбцов таблицы
    fields = data.columns.to_list()
    
    for field_name, params in fields_params.items():
        # Если имени поля нет в списке определений, переходим к следующему
        if (field_name not in fields or field_name == TARGET_FEATURE):
            continue
        
        display(Markdown(f"### Признак `{field_name}` ({params['description']})"))
        display(Markdown(f'Посмотрим на распределение признака `{field_name}` \
в разрезе целевой переменной `{TARGET_FEATURE}`.'))
        
        # Вынимаем параметры для графика
        # Копируем, чтобы не мутировать исходный словарь
        graph_params = params['distribution_by_target_graph_params'].copy()
        
        # Заголовок графика
        graph_title = graph_params.pop('title')
        
        # Если признак числовой
        if (params['field_type'] == FieldType.NUMERIC.value):
            # Выводим гистограмму и коробчатые диаграммы с распределением в разрезе целевого признака
            show_distribution_by_target(
                data, 
                field_name, 
                graph_title, 
                get_title(params['description']),
                **graph_params,
            )
            # Если признак имеет тим datetime
        elif (params['field_type'] == FieldType.DATETIME.value):
            # Выводим гистограмму в разрезе целевого признака
            show_histplot_by_target(
                data.sort_values(field_name),
                field_name,
                graph_title,
                get_title(params['description']),
                **graph_params,
            )
            # Если признак категориальный или бинарный
        elif (params['field_type'] in [FieldType.CATEGORY.value, FieldType.BINARY.value]):
            # Выводим столбчатую диаграмму в разрезе целевого признака
            show_countplot_by_target(
                data, 
                field_name, 
                graph_title, 
                get_title(params['description']),
                **graph_params,
            )
            # Выводим перевод значений категориального признака
            show_translate_feature_values(field_name)


def display_fields_correlation(
        data_source: pd.DataFrame, 
        corr_coef: float,
        method: str = 'pearson',
        should_display_corr_matrix: bool = True,
        barplot_figsize: tuple = (12, 6),
    ):
    """Выводит матрицу корреляции и пары признаков, у которых коэффициент корреляции 
    больше, чем corr_coef.
    А также столбчатую диаграмму с коэффициентами корреляции признаков с целевой переменной.

    Args:
        data_source (pd.DataFrame): таблица с данными
        corr_coef (float): от какого значения корреляции выводим пары признаков
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
        should_display_corr_matrix (bool): Выводить ли матрицу корреляции. По умолчанию True.
        barplot_figsize (tuple): Размер столбчатой диаграммы. По умолчанию (12, 6)
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    if (should_display_corr_matrix):
        display(Markdown('Построим матрицу корреляции.'))        
        # Выводим тепловую карту матрицы корреляции
        show_correlation_matrix(data, method)
    
    display(Markdown(f'Посмотрим на пары признаков с коэффициентом корреляции больше, чем `{corr_coef}`'))
    
    # Выводим пары признаков, у которых корреляция больше, чем corr_coef
    display(
        get_multicolinear_pairs(data, corr_coef, method)
    )
    
    # Выводим столбчатую диаграмму с коэффициентами корреляции признаков с целевой переменной
    show_corellation_by_target(data, method, barplot_figsize)
    
    
def display_relationship_features(
        data_source: pd.DataFrame,
        feature_x: str,
        feature_y: str = None,
        feature_target: str = TARGET_FEATURE,
        title: str = None,
    ):
    """Выводит график, иллюстрирующий зависимость переданных переменных друг от друга

    Args:
        data_source (pd.DataFrame): таблица с данными
        feature_x (str): название признака, который откладываем по оси X
        feature_y (str, optional): название признака, который откладываем по оси Y. 
                                   Может быть пустым, по умолчанию None.
        feature_target (str, optional): в разрезе какого признака нужно показать распределение. 
                                        По умолчанию - TARGET_FEATURE.
        title (str, optional): Заголовок графика. Может быть пустым, по умолчанию None.
                               Если не передан, будет сформирован из названий признаков.
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    # Если не передан признак для оси Y, но переданы остальные
    if (not feature_y and feature_x and feature_target):
        display(Markdown(f'### Взаимосвязь признаков `{feature_x}` и `{feature_target}`'))
        
        display(Markdown('Вспомним описание признаков:'))
        
        # Выведем описания признаков
        display_field_description(feature_x)
        display_field_description(feature_target)
        
        # Параметры признака, который откладываем по оси X
        params_x = fields_params[feature_x]
        
        # Вынимаем параметры для графика
        # Копируем, чтобы не мутировать исходный словарь
        graph_params = params_x['distribution_by_target_graph_params'].copy()
        
        # Заголовок графика
        graph_params.pop('title')
        
        # Если заголовок не указан, сформируем его
        if (not title):
            title = f"{get_title(params_x['description'])} в разрезе признака {feature_target}"
        
        # Если по оси X откладываем числовой признак
        if (params_x['field_type'] == FieldType.NUMERIC.value):
            # Выводим график распределения в разрезе feature_target
            # (гистограмму и коробчатые диаграммы)
            show_distribution_by_target(
                data,
                feature_x, 
                title, 
                get_title(params_x['description']),
                target_field_name=feature_target,
                **graph_params,
            )
            # Если по оси X откладываем категориальный или бинарный признак
        elif (params_x['field_type'] in [FieldType.CATEGORY.value, FieldType.BINARY.value]):
            # Выводим столбчатую диаграмму в разрезе признака feature_target
            show_countplot_by_target(
                data,
                feature_x,
                title,
                get_title(params_x['description']),
                target_field_name=feature_target,
                **graph_params,
            )
        # Если переданы все признаки (по оси X, по оси Y и в разрезе какого нужно показать различия)
    elif(feature_y and feature_x and feature_target):
        display(Markdown('### Взаимосвязь признаков `{feature_x}` и `{feature_y}` \
в разрезе признака `{feature_target}`'))
        
        display(Markdown('Вспомним описание признаков:'))
        
        # Выводим описание признаков
        display_field_description(feature_x)
        display_field_description(feature_y)
        
        # Описание целевой переменной не будем выводить много раз, мы и так помним о ней
        if (feature_target != TARGET_FEATURE):
            display_field_description(feature_target)
        
        # Параметра признака, открадываемого по оси X    
        params_x = fields_params[feature_x]
        # Параметра признака, открадываемого по оси Y 
        params_y = fields_params[feature_y]
        
        # Если заголовок не передан, сформируем его
        if (not title):
            title = f'Взаимосвязь признаков {feature_x} и {feature_y} в разрезе признака {feature_target}'

        # Выводим диаграмму рассеяния. Разным цветом красим признак feature_target
        show_scatterplot(
            data,
            feature_x,
            feature_y,
            title,
            get_title(params_x['description']),
            get_title(params_y['description']),
            feature_target,
        )
        
        
def display_best_features(selector: SelectKBest):
    """Выводит оценки признаков, полученные с помощью SelectKBest

    Args:
        selector (SelectKBest): обученный селектор SelectKBest
    """
    # Соберем DataFrame из названий признаков и их оценок
    features_data = pd.DataFrame({
        'name': selector.feature_names_in_,
        'score': selector.scores_,
    })    
    
    # Выведем таблицу с оценками признаков
    display(Markdown('**Значения оценок признаков, определенные SelectKBest**'))
    display(features_data.sort_values('score', ascending=False))
    
    # Выведем диаграмму с оценками признаков
    fig = plt.figure(figsize=(18, 7))
    barplot = sns.barplot(
        data=features_data.sort_values('score'),
        x='name',
        y='score',
    )
    barplot.set_title('Оценка признаков с помощью SelectKBest')    
    barplot.set_xlabel('Признак')
    barplot.set_ylabel('Оценка')
    barplot.tick_params(axis='x', rotation=90)
    

def display_f1_score(f1_score: float, type: str = 'test'):
    """Печатает вывод метрики F1
    Args:
        f1_score (float): Значение метрики F1
        type (str): Тип выборки ('train' - тренировочная, 'test' - тестовая)
    """
    if (type == 'test'):
        display(Markdown('f1_score на тестовом наборе: {:.4f}'.format(f1_score)))
    elif (type == 'train'):
        display(Markdown('f1_score на тренировочном наборе: {:.4f}'.format(f1_score)))


def display_optuna_info(study: optuna.Study):
    """Выводит информацию из данных optuna о подборе гиперпараметров модели

    Args:
        study (optuna.Study): объект исследования
    """
    display(Markdown(f'**Наилучшие значения гиперпараметров:**'))
    pprint(study.best_params)
    
    # График с историей оптимизации метрики
    display(optuna.visualization.plot_optimization_history(study, target_name='f1_score'))
    
    # Наиболее важные для оптимизации параметры
    display(optuna.visualization.plot_param_importances(study, target_name='f1_score'))
    
    # Словарь с наиболее значимыми для оптимизации параметрами
    importances_params = optuna.importance.get_param_importances(study)
    
    # График зон улучшения метрики для двух наиболее значимых параметров
    display(
        optuna.visualization.plot_contour(
            study, 
            params=list(importances_params.keys())[:2],
            target_name='f1_score',
        )
    )
    
    
def display_exec_time(exec_time: str):
    """Выводит время выполнения ячейки

    Args:
        exec_time (str): время выполнения
    """
    display(Markdown(f'Время выполнения: {exec_time}'))


def display_metrics_results(metrics_results: dict, optimize_times: dict):
    """Выводит таблицу с итоговой метрикой F1 для каждой модели,
    а также таблицу с временем поиска гиперпараметров для каждой модели.

    Args:
        metrics_results (dict): словарь с метриками
        optimize_times (dict): словарь со временем поиска гиперпараметров
    """
    # Построим тепловую карту с сохраненными метриками
    metrics_results_df = pd.DataFrame(metrics_results)
    metrics_results_df = metrics_results_df.set_index('model')
    sns.heatmap(metrics_results_df, annot=True, cmap='coolwarm', fmt ='.4g')
    plt.title('Тепловая карта зависимости метрики F1 от модели и способа подбора гиперпараметров')
    plt.xlabel('Cпособ подбора гиперпараметров')
    plt.ylabel('Модель')    
    
    # Выведем таблицу со временем поиска гиперпараметров
    display(Markdown('**Время поиска гиперпараметров для каждой модели**'))
    display(pd.DataFrame(optimize_times))