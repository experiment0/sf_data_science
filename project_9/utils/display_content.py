# Функции для вывода информации по исследованию таблиц и признаков

from IPython.display import display, Markdown
from IPython.core.display import HTML
import pandas as pd

from utils.constants import (
    F,
    field_semantic_category_descriptions,
    FieldOriginCategory,
    fields,
)
from utils.helpers import (
    get_multicolinear_pairs,
)
from utils.graphics import (
    dislay_correlation_matrix,
    display_corellation_by_target,
)


def display_base_fields_description(should_show_source_description: bool = False) -> None:
    """Выводит описание полей исходной таблицы, разбитое по категориям

    Args:
        should_show_source_description (bool, optional): Выводить ли описание поля с единицей измерения, 
            полученной из исходной таблицы. По умолчанию False.
    """
    display(Markdown(f'### Описание полей таблицы'))
    
    # Пройдем по списку категорий полей
    for category_name, category_description in field_semantic_category_descriptions.items():
        # Строка с описанием полей для вывода на экран
        field_descriptions = ''
        
        # Пройдем по списку данных полей таблицы
        for field_name, field_data in fields.items():
            if (
                # Выводим описания полей только для исходной таблицы
                field_data['origin_category'] == FieldOriginCategory.BASE.value and
                # Если поле относится к текущей категории
                field_data['semantic_category'] == category_name
            ):
                # Сохраним описание поля
                description = field_data['description']
                if (should_show_source_description and 'description_source' in field_data):
                    description = field_data['description_source']
                
                field_descriptions += f'<li><code>{field_name}</code> - {description}</li>'
        
        # Если собралась строка с описанием полей по заданным условиям
        if (field_descriptions):
            # Выведем название категории
            display(HTML(f'<strong>{category_description}</strong>'))
            # Выведем список с описаниями
            display(HTML(f'<ul>{field_descriptions}</ul>'))
                

def dislay_some_fields_description(
    some_fields: list,
    should_show_source_description: bool = False,
    title: str = 'Описание полей',
) -> None:
    """Выводит описание переданных полей

    Args:
        some_fields (list): поля, описание которых нужно вывести
        should_show_source_description (bool, optional): Выводить ли описание поля с единицей измерения, 
            полученной из исходной таблицы. По умолчанию False.
        title (str, optional): заголовок над списком полей.
            По умолчанию 'Описание полей'
    """
    display(Markdown(f'**{title}:**'))
    
    for field_name in some_fields:
        if (field_name in fields):
            # Выведем название и описание поля
            description = fields[field_name]['description']
            if (should_show_source_description and 'description_source' in fields[field_name]):
                description = fields[field_name]['description_source']
        else:
            description = ''
        
        display(Markdown(f'- `{field_name}` - {description}'))


def display_field_description(
    field_name: F,
    should_show_source_description: bool = False
) -> None:
    """Выводит описание поля

    Args:
        field_name (F): имя поля
        should_show_source_description (bool, optional): Выводить ли описание поля с единицей измерения, 
            полученной из исходной таблицы. По умолчанию False.
    """
    description = fields[field_name]['description']
    if (should_show_source_description and 'description_source' in fields[field_name]):
        description = fields[field_name]['description_source']
    
    display(Markdown(f'`{field_name}` - {description}'))
    
        
def display_fields_correlation(
        data_source: pd.DataFrame, 
        corr_coef: float = 0.7,
        matrix_figsize: tuple = (18, 18),
        barplot_figsize: tuple = (12, 6),
        method: str = 'pearson',
        should_display_corr_matrix: bool = True,
    ):
    """Выводит матрицу корреляции и пары признаков, у которых коэффициент корреляции 
    больше, чем corr_coef.
    А также столбчатую диаграмму с коэффициентами корреляции признаков с целевой переменной.

    Args:
        data_source (pd.DataFrame): таблица с данными
        corr_coef (float): от какого значения корреляции выводим пары признаков. По умолчанию 0.7
        matrix_figsize (tuple): Размер матрицы корреляций. По умолчанию (18, 18)
        barplot_figsize (tuple): Размер столбчатой диаграммы. По умолчанию (12, 6)
        method (str): метод рассчета корреляции. По умолчанию 'pearson'
        should_display_corr_matrix (bool): Выводить ли матрицу корреляции. По умолчанию True.
    """
    # Делаем копию, чтобы не мутировать исходные данные
    data = data_source.copy()
    
    if (should_display_corr_matrix):
        display(Markdown('Построим матрицу корреляции.'))        
        # Выводим тепловую карту матрицы корреляции
        dislay_correlation_matrix(data, matrix_figsize, method)
    
    display(Markdown(f'Посмотрим на пары признаков с коэффициентом корреляции больше, чем `{corr_coef}`'))
    
    # Выводим пары признаков, у которых корреляция больше, чем corr_coef
    display(
        get_multicolinear_pairs(data, corr_coef, method)
    )
    
    # Выводим столбчатую диаграмму с коэффициентами корреляции признаков с целевой переменной
    display_corellation_by_target(data, barplot_figsize, method)
