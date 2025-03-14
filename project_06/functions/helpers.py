# Содержит различные вспомогательные функции

import pandas as pd
from datetime import datetime
import time
from IPython.display import display, Markdown

def get_title(title: str) -> str:
    """Переводит в верхний регистр первую букву строки.

    Args:
        title (str): исходная строка

    Returns:
        str: исходная строка с первой заглавной буквой
    """
    return title[0].upper() + title[1:]


# Словарь с переводом значений признаков на русский
translate = {
    'job': {
        'management': 'менеджмент',
        'blue-collar': '«синие воротнички»',
        'technician': 'техник',
        'admin.': 'админ',
        'services': 'услуги',
        'retired': 'пенсионер',
        'self-employed': 'самозанятый',
        'student': 'студент',
        'unemployed': 'безработный',
        'entrepreneur': 'предприниматель',
        'housemaid': 'горничная',
        'unknown': 'неизвестный',
    },
    'marital': {
        'married': 'женат',
        'single': 'одинокий',
        'divorced': 'в разводе',
    },
    'education': {
        'secondary': 'среднее образование',
        'tertiary': 'высшее образование',
        'primary': 'начальное образование',
        'unknown': 'неизвестно',
    },
    'contact': {
        'cellular': 'сотовый телефон',
        'unknown': 'неизвестный тип контакта',
        'telephone': 'стационарный телефон',
    },
    'poutcome': {
        'unknown': 'неизвестный результат',
        'failure': 'неуспешный результат',
        'success': 'успешный результат',
        'other': 'другое',
    },
}


def show_translate_feature_values(field_name: str):
    """Выводит таблицу с переводом значений признака с английского на русский

    Args:
        field_name (str): имя признака, для значений которого нужно вывести перевод
    """
    if (field_name not in translate):
        return None
    
    translate_field = translate[field_name]
    
    display(Markdown(f'**Перевод значений признака `{field_name}`**'))
    
    display(
        pd.DataFrame(translate_field.items(), columns=['Значение', 'Перевод'])
    )


def get_multicolinear_pairs(data: pd.DataFrame, corr_coeff: float, method: str = 'pearson') -> list:
    """Возвращает пары мультиколинеарных признаков и значение кореляции между ними.
    Args:
        data (pd.DataFrame): таблица с данными
        corr_coeff (float): коэффициент корреляции, выше которого отбираем пары
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
    Returns:
        list: Список с именами столбцов и коэффициентом корреляции
              ([столбец1, столбец2], коэффициент)
    """
    # Матрица корреляции признаков
    data_corr = data.corr(method=method)
    
    # Имена колонок таблицы
    col_names = list(data.columns)
    
    # Соберем в этот список пары колонок без повторений
    colls_pairs = []
    
    for col1 in col_names:
        for col2 in col_names:
            if (col1 != col2 and [col1, col2] not in colls_pairs and [col2, col1] not in colls_pairs):
                colls_pairs.append([col1, col2])
    
    # Соберем в этот список пары признаков, 
    # у которых коэффициент корреляции больше corr_coeff
    multicolinear_pairs = []
    
    # Перебираем все пары признаков без повторений
    for pair in colls_pairs:
        # Вынимаем коэффициент корреляции признаков из матрицы корреляции
        corr_between_colls = round(data_corr[pair[0]][pair[1]], 2)
        # Если по модулю коэффициент корреляции больше
        if (abs(corr_between_colls) > abs(corr_coeff)):
            # Добавляем пару в наш список
            multicolinear_pairs.append((pair, corr_between_colls))
            
    return multicolinear_pairs


def get_year_xticks() -> list:
    """Возвращает метки по оси X для меток признака с типом datetime за год

    Returns:
        list: список меток
    """
    xticks = []
    
    # Возьмем текущий год для формирования даты
    year = datetime.now().year
    
    for month in range(1, 13):
        month_digit = str(month)
        if (month < 10):
            month_digit = '0' + str(month)
        xticks.append(f'{year}-{month_digit}')
    
    return xticks 


def get_exec_time(start: float, end: float) -> str:
    """Возвращает время выполнения в секундах

    Args:
        start (float): время начала
        end (float): время окончания

    Returns:
        str: строка с форматированным временем выполнения
    """
    duration = end - start
    formatted_duration = time.strftime("%H:%M:%S", time.gmtime(duration))
    return formatted_duration