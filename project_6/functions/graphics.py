# Содержит функции для вывода графиков

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from abc import ABCMeta

from functions.constants import TARGET_FEATURE, TITLE_FONT_SIZE, RANDOM_STATE, MAX_ITER

def show_describe(
        describe: pd.Series, 
        field_name: str = None, 
        title: str = None,
    ):
    """Выводит таблицу с числовыми характеристиками распределения признака

    Args:
        describe (pd.Series): характеристики распределения
        field_name (str, optional): имя поля, для которого выводятся характеристики. По умолчанию None.
        title (str, optional): заголовок над таблицей. По умолчанию None.
    """
    if (field_name and not title):
        title = f'Числовые характерстики распределения значений признака `{field_name}`'
    
    display(Markdown(f'**{title}**'))

    display(pd.DataFrame({'Характеристика': describe.index, 'Значение': describe.values}))


def show_distribution(
        data: pd.DataFrame, 
        field_name: str, 
        title: str, 
        xlabel: str, 
        ylabel: str = 'Число клиентов',
    ):
    """Выводит гистограмму, коробчатую диаграмму и числовые характеристики
    расределения признака field_name

    Args:
        data (pd.DataFrame): таблица с данными
        field_name (str): имя колонки, для которой нужно вывести распределение
        title (str): заголовок графика
        xlabel (str): подпись по оси X
        ylabel (str, optional): Подпись по оси Y. По умолчанию: 'Число клиентов'.
    """
    # Задаем фигуру и координатную плоскость
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    
    # Задаем нулевое расстояние между верхним и нижним графиком
    plt.subplots_adjust(hspace=0)
    
    # Параметры гистограммы
    histplot = sns.histplot(
        data=data, 
        x=field_name, 
        kde=True,
        bins=42,
        ax=axes[0],
    )
    histplot.set_title(title, fontsize=TITLE_FONT_SIZE)    
    histplot.set_ylabel(ylabel)
    
    # Среднее и медиана на гистограмме
    field_values = data[field_name]
    histplot.axvline(field_values.mean(), color='red', lw=2, label='Среднее')
    histplot.axvline(field_values.median(), color='red', lw=2, label='Медиана', ls='--')
    histplot.legend()
    
    # Параметры коробчатой диаграммы
    boxplot = sns.boxplot(
        data=data,
        x=field_name,
        ax=axes[1],
    )
    boxplot.set_xlabel(xlabel)
    boxplot.grid()
    
    # Выведем график
    plt.show()
    
    # Выведем числовые характеристики распределения
    show_describe(
        data[field_name].describe(),
        field_name,
    )
    

def show_countplot(
        data: pd.DataFrame, 
        field_name: str, 
        title: str, 
        xlabel: str, 
        ylabel: str = 'Количество клиентов', 
        x_rotation:int = 0, 
        should_sort: bool = True,
        figsize: tuple = (16, 8),
    ):
    """Выводит столбчатую диагармму и характеристики распределения

    Args:
        data (pd.DataFrame): таблица с данными
        field_name (str): имя признака, для которого нужно вывести диаграмму
        title (str): заголовок графика
        xlabel (str): подпись по оси X
        ylabel (str, optional): подпись по оси Y. По умолчанию 'Количество клиентов'
        x_rotation (int, optional): угол поворота подписи по оси X. По умолчанию 0 градусов.
        should_sort (bool, optional): нужно ли сортировать столбцы диаграммы по убыванию. По умолчанию True.
        figsize (tuple, optional): кортеж с размерами фигуры. По умолчанию (16, 8)
    """
    # Если необходимо, задаем сортировку столбцов по убыванию количества значений в них
    order = None
    if (should_sort):
        order = data[field_name].value_counts().index
    
    # Размер фигуры
    plt.figure(figsize=figsize)
    
    # Поворот меток по оси X
    plt.xticks(rotation=x_rotation)
    
    # Параметры столбчатой диаграммы
    ax = sns.countplot(x=data[field_name], order=order)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    # Выведем график
    plt.show()
    
    # Выведем числовые характеристики распределения
    show_describe(
        data[field_name].astype('str').describe(include='object'),
        field_name,
    )
    

def show_pie(data: pd.DataFrame, field_name: str, title: str):
    """Выводит круговую диаграмму

    Args:
        data (pd.DataFrame): таблица с данными
        field_name (str): имя признака, для которого нужно вывести диаграмму
        title (str): название графика
    """
    # Фигура и координатная плоскость
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes((1, 1, 1, 1))
    
    # Количество значений по каждой категории
    counts = data[field_name].value_counts().sort_values(ascending=False)
    
    # Параметры круговой диаграммы
    pie = ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.2f%%',
        startangle=90
    )
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    
    # Выведем график
    plt.show(pie)
    

def show_countplot_by_target(
        data: pd.DataFrame, 
        field_name: str, 
        title: str, 
        xlabel: str, 
        ylabel: str = 'Количество клиентов', 
        target_field_name: str = TARGET_FEATURE,
        x_rotation: int = 0, 
        should_sort: bool = True,
        figsize: tuple = (16, 8),
    ):
    """Выводит столбчатую диагармму в разрезе заданного признака (target_field_name)

    Args:
        data (pd.DataFrame): таблица с данными
        field_name (str): имя признака, для которого нужно вывести диаграмму
        title (str): заголовок графика
        xlabel (str): подпись по оси X
        ylabel (str, optional): подпись по оси Y. По умолчанию 'Количество клиентов'.
        target_field_name (str, optional): имя целевой переменной. По умолчанию TARGET_FEATURE
        x_rotation (int, optional): угол поворота подписи по оси X. По умолчанию 0 градусов.
        should_sort (bool, optional): нужно ли сортировать столбцы диаграммы по убыванию. По умолчанию True.
        figsize (tuple, optional): кортеж с размерами фигуры. По умолчанию (16, 8)
    """
    # Если необходимо, задаем сортировку столбцов по убыванию количества значений в них
    order = None
    if (should_sort):
        order = data[field_name].value_counts().index
    
    # Фигура
    plt.figure(figsize=figsize)
    
    # Параметры столбчатой диаграммы
    ax = sns.countplot(
        data=data, 
        x=data[field_name], 
        hue=target_field_name, 
        order=order
    )
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    # Поворот меток по оси X
    plt.xticks(rotation=x_rotation)
    
    # Выводим график
    plt.show()
    
    
def show_histplot_by_target(
        data: pd.DataFrame, 
        field_name: str, 
        title: str, 
        xlabel: str, 
        ylabel: str = 'Количество клиентов', 
        x_rotation: int = 0, 
        xticks: range = None,
        binwidth: float = None,
    ):
    """Выводит гистограмму в разрезе целевого признака

    Args:
        data (pd.DataFrame): таблица с данными
        field_name (str): имя признака, для которого нужно вывести гистограмму
        title (str): заголовок графика
        xlabel (str): подпись по оси X
        ylabel (str, optional): подпись по оси Y. По умолчанию 'Количество клиентов'.
        x_rotation (int, optional): угол поворота подписи по оси X. По умолчанию 0 градусов.
        xticks (range, optional): range(начало, конец, шаг) - метки по оси X. По умолчанию None
        binwidth (float, optional): параметр binwidth для гистограммы (ширина корзины). По умолчанию None
    """
    # Фигура
    fig = plt.figure(figsize=(16, 8))
    
    # Параметры гистограммы
    ax = sns.histplot(
        data=data, 
        x=field_name, 
        hue=TARGET_FEATURE, 
        alpha=1, 
        multiple='stack', 
        binwidth=binwidth,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    
    # Поворот меток по оси X
    plt.xticks(rotation=x_rotation)
    
    # Шаг меток по оси X
    if (xticks):
        ax.set_xticks(xticks)

    # Выведем график
    plt.show()


def show_distribution_by_target(
        data: pd.DataFrame, 
        field_name: str, 
        title: str, 
        xlabel: str, 
        ylabel: str = 'Количество клиентов', 
        xticks: range = None,
        binwidth: float = None,
        target_field_name: str = TARGET_FEATURE,
    ):
    """Выводит гистограмму и коробчатые диаграммы в разрезе переданного целевого признака (target_field_name)

    Args:
        data (pd.DataFrame): таблица с данными
        field_name (str): имя признака, для которого нужно вывести распределение
        title (str): заголовок графика
        xlabel (str): подпись по оси X
        ylabel (str, optional): подпись по оси Y. По умолчанию 'Количество клиентов'.
        xticks (range, optional): range(начало, конец, шаг) - метки по оси X. По умолчанию None
        binwidth (float, optional): параметр binwidth для гистограммы (ширина корзины). По умолчанию None
        target_field_name (str, optional): в разрезе какого признака строим распределение. По умолчанию TARGET_FEATURE
    """
    # Задаем фигуру и координатную плоскость
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))
    
    # Параметры гистограммы
    histplot = sns.histplot(
        ax=axes[0],
        data=data, 
        x=field_name, 
        hue=target_field_name,
        alpha=1, 
        multiple='stack', 
        binwidth=binwidth,        
    )
    histplot.set_title(title, fontsize=TITLE_FONT_SIZE)
    histplot.set_xlabel(xlabel)  
    histplot.set_ylabel(ylabel)

    # Задаем шаг меток по оси X
    if (xticks):
        histplot.set_xticks(xticks)
        
    # Параметры коробчатой диаграммы
    boxplot = sns.boxplot(
        ax=axes[1],
        data=data,
        x=field_name,
        y=target_field_name,
        orient='h',
    )
    boxplot.set_xlabel(xlabel)
    boxplot.grid()
    
    # Выведем график
    plt.show()


def show_correlation_matrix(data: pd.DataFrame, method: str = 'pearson'):
    """Выводит матрицу корреляции

    Args:
        data (pd.DataFrame): таблица с данными
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
    """
    # Задаем фигуру и координатную плоскость
    fig, axes = plt.subplots(figsize=(18, 18))
    
    # Параметры тепловой карты
    sns.heatmap(
        data.corr(method=method), 
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
    

def show_corellation_by_target(data: pd.DataFrame, method: str = 'pearson', figsize: tuple = (12, 6)):
    """Выводит столбчатую диаграмму корреляции признаков с целевой переменной

    Args:
        data (pd.DataFrame): таблица с данными
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
        figsize (tuple): размер графика. По умолчанию (12, 6)
    """
    # Матрица корреляции
    data_corr = data.corr(method=method)

    # Сортируем колонку целевого признака по модулю
    target_corr_sorted = data_corr[TARGET_FEATURE].abs().sort_values()

    # Сделаем из столбца таблицу
    target_corr_df = pd.DataFrame({
        'feature': target_corr_sorted.index, 
        'coef_corr_abs': target_corr_sorted.values, 
    })

    # Построим столбчатую диаграмму
    plt.figure(figsize=figsize)
    plt.xticks(rotation=90)
    
    barplot = sns.barplot(data=target_corr_df, x='feature', y='coef_corr_abs')
    barplot.set_title(
        f'Корреляция признаков с целевой переменной {TARGET_FEATURE}', 
        fontsize=TITLE_FONT_SIZE
    )
    barplot.set_xlabel('Признак')
    barplot.set_ylabel(f'Коэффициент корреляции с {TARGET_FEATURE}')
    
    # Выведем график
    plt.show()
    
    
def show_scatterplot(
        data: pd.DataFrame, 
        x: str, 
        y: str, 
        title: str,
        xlabel: str,
        ylabel: str,
        hue: str = TARGET_FEATURE,
    ):
    """Выводит диаграмму рассеяния в разрезе целевого признака

    Args:
        data (pd.DataFrame): таблица с данными
        x (str): имя признака, откладываемого по оси X
        y (str): имя признака, откладываемого по оси Y
        title (str): заголовок графика
        xlabel (str): подпись по оси X
        ylabel (str): подпись по оси Y
        hue (str, optional): какой признак красим в разные цвета. По умолчанию TARGET_FEATURE
    """
    # Фигута
    fig = plt.figure(figsize=(18, 10))

    # Параметры диаграммы рассеяния
    scatterplot = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
    )    
    scatterplot.set_title(title, fontsize=TITLE_FONT_SIZE)
    scatterplot.set_xlabel(xlabel)
    scatterplot.set_ylabel(ylabel)
    
    # Выведем график
    plt.show()


def show_dependence_of_model_parameters(
        Model: ABCMeta,        
        X_train: np.ndarray,
        X_test: np.ndarray, 
        y_train: pd.Series, 
        y_test: pd.Series,
        estimators: list = [],
        FinalStackingModel: ABCMeta = DecisionTreeClassifier,
    ):
    """Выводит график соответствий параметра модели
    и соответствующей ему метрики F1.
    Для тестовой и тренировочной выборок

    Args:
        Model (ABCMeta): модель, для которой подбираем значение max_depth
        X_train (np.ndarray): тренировочная выборка
        X_test (np.ndarray): тестовая выборка
        y_train (pd.Series): ответы тренировочной выборки
        y_test (pd.Series): ответы тестовой выборки
        estimators (list): список моделей для стекинга
        FinalStackingModel (ABCMeta): финальная модель для стекинга
    """
    # Название модели
    model_name = Model.__name__
    # Флаг, что имеем дело с логистической регрессией
    is_logistic_regression = model_name == 'LogisticRegression'
    # Флаг, что мы имеем дело со стекингом
    is_stacking = model_name == 'StackingClassifier'
    # Если модель - не логистическая регрессия и не стекинг, значит мы имеем дело с деревом
    
    # Создаём список из возможных значений
    if (is_logistic_regression):
        # Параметр C
        param_list = np.arange(0.01, 3.1, 0.1)
        param_name = 'C'
    elif (is_stacking):
        # Параметр max_depth
        param_list = np.arange(1, 6, 1)
        param_name = 'max_depth'
    else:
        # Параметр max_depth
        param_list = np.arange(1, 21, 1)
        param_name = 'max_depth'

    # Создаём пустые списки, в которые будем добавлять результаты 
    train_scores = []
    test_scores = []

    for param_value in param_list:
        # Создаём модель
        if (is_logistic_regression):
            model = Model(                
                random_state=RANDOM_STATE, 
                max_iter=MAX_ITER,
                C=param_value,
            )
        elif (is_stacking):
            model = Model(
                estimators=estimators,
                final_estimator=FinalStackingModel(
                    random_state=RANDOM_STATE,
                    max_depth=param_value,
                )
            )
        else:
            model = Model(
                random_state=RANDOM_STATE,
                max_depth=param_value,
            )
            
        # Обучаем модель
        model.fit(X_train, y_train)
        
        # Делаем предсказание для тренировочной выборки
        y_train_pred = model.predict(X_train)
        
        # Делаем предсказание для тестовой выборки
        y_test_pred = model.predict(X_test)
        
        # Рассчитываем коэффициенты детерминации для двух выборок и добавляем их в списки
        train_scores.append(metrics.f1_score(y_train, y_train_pred))
        test_scores.append(metrics.f1_score(y_test, y_test_pred))
    
    # Визуализируем изменение F1 в зависимости от значения параметра

    # Фигура + координатная плоскость
    fig, ax = plt.subplots(figsize=(12, 4)) 

    # Линейный график для тренировочной выборки
    ax.plot(param_list, train_scores, label='Train') 

    # Линейный график для тестовой выборки
    ax.plot(param_list, test_scores, label='Test') 

    ax.set_title(f'Зависимость метрики F1 от параметра {param_name}')

    # Название оси абсцисс
    ax.set_xlabel(param_name) 

    # Название оси ординат
    ax.set_ylabel('f1 score') 

    # Метки по оси абсцисс
    ax.set_xticks(param_list) 

    # Поворот меток на оси абсцисс
    ax.xaxis.set_tick_params(rotation=45) 

    # Отображение легенды
    ax.legend(); 