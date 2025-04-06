# Функции для построения графиков

import numpy as np
import pandas as pd
import statsmodels.graphics.tsaplots as sgt
from matplotlib import pyplot as plt


def display_corelogram(series: pd.Series, lags_count: int = 26) -> None:
    """Выводит графики автокорреляции и частичной автокорреляции

    Args:
        series (pd.Series): столбец со значениями ряда
        lags_count (int, optional): количество лагов на графике.
            По умолчанию 26.
    """
    # Фигура и координатная плоскость (задаем сетку в одну строку и 2 столбца)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    # График автокорреляции
    sgt.plot_acf(
        series, 
        ax=axes[0], 
        lags=lags_count, 
        title='Автокорреляция (определяем q)',
    )
    # Метки по оси OX
    axes[0].set(xticks = np.arange(lags_count+1))
    # Сетка
    axes[0].grid()
    
    # График частичной автокорреляции
    sgt.plot_pacf(
        series, 
        ax=axes[1], 
        lags=lags_count, 
        method='ywm', 
        title='Частичная автокорреляция (определяем p)'
    )
    # Метки по оси OX
    axes[1].set(xticks = np.arange(lags_count+1))
    axes[1].grid()
    
    plt.show()


def display_prediction(
    source_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    predict: pd.Series, 
    forecast_data: pd.DataFrame,
) -> None:
    """Выводит график с данными результата предсказания

    Args:
        source_data (pd.DataFrame): исходная таблица с истинными значениями
        test_data (pd.DataFrame): тестовая выборка
        predict (pd.Series): предсказание
        forecast_data (pd.DataFrame): таблица с доверительными интервалами
    """
    # Сначала соберем в таблицу graph_data все данные для графика
    # (истинные и предсказанные значения, 
    # а также значения верхней и нижней границ 95%-ого доверительного интервала для предсказанных значений)
    
    # Устанавливаем индексы с датами исходной таблицы
    graph_data = pd.DataFrame(index=source_data.index)

    # Столбец с предсказанными значениями
    graph_data['predict'] = predict

    # Столбец с исходными истинными значениями
    graph_data['true_values'] = source_data['val']

    # Добавляем столбцы со значениями нижней и верхней границ
    # доверительного интервала
    graph_data.at[test_data.index, 'conf_int_lower'] = forecast_data['lower val']
    graph_data.at[test_data.index, 'conf_int_upper'] = forecast_data['upper val']
    
    # Теперь отобразим все данные на графике

    # Размер фигуры
    plt.rcParams['figure.figsize'] = 18, 12

    # Задаем сетку
    plt.subplot(
        2, # количество строк в сетке
        1, # количество столбцов в сетке
        1, # индексное положение участка на сетке, в котором будем выводить график
    )

    # График с истинными значениями
    plt.plot(
        graph_data['true_values'], # все истинные значения
        color='blue', # синий цвет графика
        label='Истинные значения', # название метки
        alpha=0.4, # прозрачность графика
    )

    # График с предсказаниями
    plt.plot(
        # Берем только индексы тестовой таблицы
        # и красным цветом обозначаем предсказания
        graph_data.loc[test_data.index]['predict'], # индексы и значения, которые рисуем
        color='red', # красный цвет
        linestyle='-', # стиль линии
        label='Предсказанные значения', # название метки
    )

    # Нижнее значение доверительного интервала
    plt.plot(
        # Берем только индексы тестовой таблицы
        # и зеленым цветом обозначаем линию нижнего доверительного интервала для предсказания
        graph_data.loc[test_data.index]['conf_int_lower'],
        color='green', # цвет графика - зеленый
        linestyle='--', # стиль линии
        label='Доверительный интервал (95%)', # название метки
        alpha=0.4, # прозрачность линии
    )

    # Верхнее значение доверительного интервала
    plt.plot(
        graph_data.loc[test_data.index]['conf_int_upper'],
        color='green',
        linestyle='--',
        alpha=0.4,
    )

    # Название графика 
    plt.title('Результат предсказания суммы продаж в сфере розничной торговли и услуг общественного питания в США', pad=20)
    plt.xlabel('Год и месяц')
    plt.ylabel('Млн. $ США')

    # Отображаем легенду
    plt.legend()
    # Отображаем сетку
    plt.grid()

    # Теперь рисуем увеличенную часть с предсказанием

    # Задаем сетку
    plt.subplot(
        2, # количество строк в сетке
        2, # количество столбцов в сетке отрисовки
        3, # задает индексное положение участка на сетке. 
        # Индекс начинается с 1 в левом верхнем углу и увеличивается вправо.
    )

    # Истинные значения с индексами из тестовой выборки
    plt.plot(
        graph_data.loc[test_data.index]['true_values'], # индексы и их значения
        color='blue', # цвет графика
        label='Истинные значения', # название метки
        alpha=0.6, # прозрачность линии
    )

    # Предсказанные значения
    plt.plot(
        graph_data.loc[test_data.index]['predict'],
        color='red',
        linestyle='-',
        label='Предсказанные значения',
        alpha=0.6,
    )

    # Нижняя линия доверительного интервала
    plt.plot(
        graph_data.loc[test_data.index]['conf_int_lower'],
        color='green',
        linestyle='--',
        label='Доверительный интервал (95%)',
        alpha=0.6,
    )

    # Верхняя линия доверительного интервала
    plt.plot(
        graph_data.loc[test_data.index]['conf_int_upper'],
        color='green',
        linestyle='--',
        alpha=0.6,
    )
    
    # Подписи к графику
    plt.title('Результат предсказания продаж на фоне тестовой выборки', pad=20)
    plt.xlabel('Год и месяц')
    plt.ylabel('Млн. $ США')
    plt.xticks(test_data.index, rotation=45)

    # Выводим легенду
    plt.legend()
    # Выводим сетку
    plt.grid()

    # Выравниваем графики
    plt.tight_layout()

    # Выводим график
    plt.show();