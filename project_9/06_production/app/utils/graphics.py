# Функции для построения графиков

import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import F


def create_prediction_graphic(
    location_past_data: pd.DataFrame,
    location_future_data: pd.DataFrame,
) -> None:
    """Строит график со значениями ожидаемой продолжительности жизни страны
       для предыдущих лет и для прогноза.

    Args:
        location_past_data (pd.DataFrame): таблица с данными страны за прошлые года
        location_future_data (pd.DataFrame): таблица с предикторами и прогнозом для страны
    """
    # Название страны
    location = location_future_data[F.Location.value][0]
    
    # Задаем размер графика
    plt.figure(figsize=(14, 6))
    
    # График значений за прошлые года
    (line_past,) = plt.plot(
        location_past_data.set_index(F.Period.value)[F.LifeExpectancy.value],    
        color="blue",
        marker="o", 
    )
    # График прогноза
    (line_future,) = plt.plot(
        location_future_data.set_index(F.Period.value)[F.LifeExpectancy.value],    
        color="red",
        marker="o", 
    )
    
    # Задаем вывод подписей к линиям графиков
    legend_lines = [line_past, line_future]
    legend_names = ['Прошлые данные', 'Прогноз']
    
    plt.legend(
        legend_lines, 
        legend_names,
        loc='upper left',
    )
    
    # Метки по оси X сделаем для каждого года
    xticks = list(location_past_data[F.Period.value].values) + \
        list(location_future_data[F.Period.value].values)
    plt.xticks(xticks, rotation=45)
    
    # Подписи к осям
    plt.xlabel('Год')
    plt.ylabel('Продолжительность жизни');
    plt.title(f'Прогноз ожидаемой продолжительности жизни для страны {location}')
    
    # Задаем сетку
    plt.grid()
    
    # Сохраняем картинку
    plt.savefig('./static/prediction.png');