# Вспомогательные функции

from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Union, Tuple


def print_stat_test_result(p_value: float, alpha: float) -> None:
    """Печатает результаты статистического теста

    Args:
        p_value (float): полученное в результате теста p-value
        alpha (float): заранее заданный уровень значимости
    """
    print('p-value: ', round(p_value, 3))
    
    if (p_value <= alpha):
        print('Отвергаем нулевую гипотезу в пользу альтернативной.')
    else:
        print('У нас нет оснований отвергнуть нулевую гипотезу.')
        

def print_shapiro_test_result(p_value: float, alpha: float, group_name: str) -> None:
    """Выводит результат теста Шапиро-Уилка на нормальное распределение

    Args:
        p_value (float): полученное в результате теста p-value
        alpha (float): заранее заданный уровень значимости
        group_name (str): название группы (A или B)
    """
    print(f'p-value группы {group_name}:', round(p_value, 2))
    
    if p_value <= alpha:
        print('Отвергаем нулевую гипотезу в пользу альтернативной.', \
            f'Распределение в группе {group_name} отлично от нормального.')
    else:
        print('Принимаем нулевую гипотезу.', \
            f'Распределение в группе {group_name} является нормальным.')
    
    print()


def print_confidence_interval(lower: float, upper: float, rounding_accuracy: int = 2) -> None:
    """Выводит границы доверительного интервала

    Args:
        lower (float): нижняя граница
        upper (float): вернхяя граница
        rounding_accuracy (int, optional): точность округления. По умолчанию 2.
    """
    print('Доверительный интервал: {}'
          .format((round(lower, rounding_accuracy), round(upper, rounding_accuracy))))
    

def display_confidence_intervals(
    a_lower: float, a_upper: float,
    b_lower: float, b_upper: float,
    title: str,
    xlabel: str,
) -> None:
    """Выводит графики доверительных интервалов

    Args:
        a_lower (float): нижняя граница доверительного интервала группы А
        a_upper (float): верхняя граница доверительного интервала группы А
        b_lower (float): нижняя граница доверительного интервала группы B
        b_upper (float): верхняя граница доверительного интервала группы B
        title (str): заголовок графика
        xlabel (str): подпись оси OX
    """
    # Фигура + координатная плоскость
    fig, ax = plt.subplots(figsize=(8, 3)) 

    # Доверительный интервал для группы A
    ax.plot(
        [a_lower, (a_lower+a_upper)/2, a_upper], 
        [0, 0, 0], 
        marker='o', 
        color='blue', 
        label='группа A',
    )
    # Доверительный интервал для группы B
    ax.plot(
        [b_lower, (b_lower+b_upper)/2, b_upper], 
        [1, 1, 1], 
        marker='o', 
        color='orange', 
        label='группа B',
    )

    # Подписи к графику
    ax.set_title(title)
    ax.set_xlabel(xlabel)
        
    # Легенда
    ax.legend(facecolor='white', fontsize=11);


def get_tour_type(price: int) -> Union[str, None]:
    """Определяет тип тура по его цене

    Args:
        price (int): цена тура

    Returns:
        Union[str, None]: тип тура
    """
    if (price == 100000):
        return 'Thailand'
    if (price == 60000):
        return 'Türkiye'
    if (price == 200000):
        return 'Maldives'
    if (price == 10000):
        return 'Saint Petersburg'
    if (price == 150000):
        return 'Kamchatka'


def get_confidence_interval_of_difference_conversions(
    x_p: list, 
    n: list, 
    gamma: float = 0.95
) -> Tuple[float, float]:
    """Рассчитывает и возвращает кортеж из вычисленных границ доверительного интервала разницы конверсий

    Args:
        x_p (list): список из выборочных пропорций для групп А и B соответственно
        n (list): список из размеров выборки для групп А и B соответственно
        gamma (float, optional): уровень надёжности. По умолчанию 0.95.

    Returns:
        Tuple[float, float]: кортеж из вычисленных границ доверительного интервала разницы конверсий
    """
    # Уровень значимости
    alpha = 1 - gamma 
    # Выборочная разница конверсий групп B и A
    diff = x_p[1] - x_p[0] 
    # z-критическое
    z_crit = -norm.ppf(alpha/2) 
    # Погрешность
    eps = z_crit * (x_p[0] * (1 - x_p[0])/n[0] + x_p[1] * (1 - x_p[1])/n[1]) ** 0.5 
    # Левая (нижняя) граница
    lower_bound = diff - eps 
    # Правая (верхняя) граница
    upper_bound = diff + eps 
    
    # Возвращаем кортеж из  границ интервала
    return lower_bound, upper_bound