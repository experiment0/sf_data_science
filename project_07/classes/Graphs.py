import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_selection
from typing import Union
from IPython.display import display

# Вспомогательные функции общего назначения
from helpers.common import get_multicolinear_pairs


class Graphs:
    """Вспомогательный класс для построения графиков
    """
    def __init__(self) -> None:       
        plt.style.use('seaborn')
        
        font = {'family': 'monospace', 'weight': 'normal', 'size': 18}
        plt.rc('font', **font)
    
    
    def show_pie(self, data: pd.DataFrame, field_name: str, title: str):
        """Выводит круговую диаграмму распределения признака

        Args:
            data (pd.DataFrame): таблица с данными
            field_name (str): имя признака, для которого выводим распределение
            title (str): заголовок графика
        """
        counts = data[field_name].value_counts()

        fig = plt.figure(figsize=(5, 5))
        axes = fig.add_axes([0, 0, 1, 1])

        axes.set_title(title)
        axes.pie(
            counts,
            labels=counts.index,
            autopct='%.1f%%',
            textprops={'fontsize': 12},
        );
    
    
    def show_countplot(self, data: pd.DataFrame, field_name: str, title: str, x_label: str, y_label: str):
        """Выводит количественную столбчатую диаграмму распределения признака

        Args:
            data (pd.DataFrame): таблица с данными
            field_name (str): имя признака, для которого выводим распределение
            title (str): название графика
            x_label (str): подпись по оси Х
            y_label (str): подпись по оси Y
        """
        ax = sns.countplot(data=data, x=data[field_name])
        plt.title(title)
        ax.set(xlabel=x_label, ylabel=y_label)
        plt.show()
        
        
    def show_barplot(self, data: pd.Series, title: str, x_label: str, y_label: str):
        """Выводит столбчатую диаграмму значений признака

        Args:
            data (pd.Series): столбец с данными, по которым будем строить диаграмму
            title (str): заголовок графика
            x_label (str): _description_
            y_label (str): _description_
        """
        ax = sns.barplot(x=data.index, y=data.values)
        plt.title(title)
        ax.set(xlabel=x_label, ylabel=y_label)
        plt.show()
    
    
    def show_barplot_of_feature_scores(
        self, 
        feature_names: pd.Series, 
        feature_values: pd.Series,
        algorithm_name: str = 'SelectKBest',
    ):
        """Выводит диаграмму значимости признаков 
        (график имеет особые постоянные настройки, поэтому вынесен в отдельную функцию)

        Args:
            feature_names (pd.Series): названия признаков
            feature_values (pd.Series): значимость признаков
            (str, optional): подпись по оси Y. По умолчанию 'SelectKBest'.
        """
        fig, ax = plt.subplots(figsize=(22, 8)) 
        barplot = sns.barplot(x=feature_names, y=feature_values)
        barplot.set_title(f'Диаграмма значимости признаков, определенная с помощью {algorithm_name}')
        barplot.set_xlabel('Название признака')
        barplot.set_ylabel('Значимость признака')
        barplot.xaxis.set_tick_params(rotation=90)
        plt.show()
        
    
    def show_histplot(
        self, 
        data: pd.DataFrame, 
        field_name: str, 
        title: str, 
        x_label: str, 
        y_label: str = 'Количество',
        hue_field_name: Union[str, None] = None,
    ):
        """Выводит гистограмму распределения признака

        Args:
            data (pd.DataFrame): таблица с данными
            field_name (str): имя признака, для которого будем строить гистограмму
            title (str): название графика
            x_label (str): подпись по оси Х
            y_label (str, optional): подпись по оси Y. По умолчанию 'Количество'.
            hue_field_name: (str | None, optional): Поле, по которому разделяем визуализацию. По умолчанию None.
        """
        fig, ax = plt.subplots(figsize=(12, 6)) 
        histplot_params = {
            'data': data,
            'x': field_name,
            'bins': 40,
            'ax': ax,
        }
        if (hue_field_name):
            histplot_params['hue'] = hue_field_name
            histplot_params['alpha'] = 0.5
        histplot = sns.histplot(**histplot_params)
        histplot.set_title(title)
        histplot.set_xlabel(x_label)
        histplot.set_ylabel(y_label)
   
    
    def show_boxplot(
        self, 
        data: pd.DataFrame, 
        field_name: str, 
        title: str, 
        x_label: str,
        y_label: Union[str, None] = None,
        hue_field_name: Union[str, None] = None,
    ):
        """Строит коробчатую диаграмму распределения признака

        Args:
            data (pd.DataFrame): таблица с данными
            field_name (str): имя признака, для которого будем строить коробчатую диаграмму
            title (str): название графика
            x_label (str): подпись по оси Х
            y_label (str | None, optional): подпись по оси Y. По умолчанию None.
            hue_field_name: (str | None, optional): Поле, по которому разделяем визуализацию. По умолчанию None.
        """
        fig, ax = plt.subplots(figsize=(12, 6)) 
        boxplot_params = {
            'data': data,
            'x': field_name,
            'orient': 'h',
            'ax': ax,
        }
        if (hue_field_name):
            boxplot_params['y'] = hue_field_name
        sns.boxplot(**boxplot_params)
        ax.set_title(title) 
        ax.set_xlabel(x_label) 
        if (y_label):
            ax.set_ylabel(y_label)
            
    
    def show_heatmap(
        self, 
        data: pd.DataFrame, 
        title: str, 
        x_label: Union[str, None] = None, 
        y_label: Union[str, None] = None,
        is_corr_matrix: bool = False,
        fmt: Union[str, None] = '.1f'
    ):
        """Выводит тепловую карту

        Args:
            data (pd.DataFrame): таблица, для которой нужно вывести тепловую карту
            title (str): название графика
            x_label (str | None, optional): подпись по оси Х. По умолчанию None.
            y_label (str | None, optional): подпись по оси Y. По умолчанию None.
            is_corr_matrix (bool, optional): строим ли мы тепловую карту для матрицы корреляции. 
                                             По умолчанию False
            fmt (str | None, optional): отображаемая точность значений. По умолчанию '.1f'.
        """
        fig, ax = plt.subplots(figsize=(22, 11))
        heatmap_params = {
            'data': data,
            'annot': True,
            'annot_kws': { 'fontsize': 10 },
            'linewidths': 0.9,
            'ax': ax,
            'fmt': fmt,
            'cmap': 'coolwarm',
        }
        if (is_corr_matrix):
            heatmap_params['vmin'] = -1
            heatmap_params['vmax'] = 1
            heatmap_params['center'] = 0
            heatmap_params['fmt'] = '.3f'
            
        sns.heatmap(**heatmap_params)
        plt.title(title)
        if (x_label and y_label):
            ax.set(xlabel=x_label, ylabel=y_label)
        plt.show()
    
    
    def show_scatterplot(
        self, 
        data: pd.DataFrame, 
        x_field_name: str,
        y_field_name: str,
        hue_field_name: str,
        title: str,
        x_label: str,
        y_label: str,
        palette: dict,
    ):
        """Выводит диаграмму рассеяния

        Args:
            data (pd.DataFrame): таблица с данными, по которым строим диаграмму
            x_field_name (str): имя поля, значения которого будем откладывать по оси Х
            y_field_name (str): имя поля, значения которого будем откладывать по оси Y
            hue_field_name (str): имя поля, в зависимости от значения которого красим точки цветом
            title (str): заголовок графика
            x_label (str): подпись по оси Х
            y_label (str): подпись по оси Y
            palette (dict): палитра
        """
        fig, ax = plt.subplots(figsize=(22, 11))
        sns.scatterplot(
            data=data,
            x=x_field_name,
            y=y_field_name,
            hue=hue_field_name,
            ax=ax,
            s=10,
            palette=palette,
            legend='full',
        )
        ax.set_title(title)
        ax.set_xlabel(x_label) 
        ax.set_ylabel(y_label);
    
    
    def show_best_and_multicolinear_features(self, X: pd.DataFrame, y: pd.Series):
        """Выводит диаграмму значимости признаков и пары мультиколлинеарных признаков

        Args:
            X (pd.DataFrame): предикторы
            y (pd.Series): целевая переменная
        """
        # Количество предикторов
        features_count = X.shape[1]
        
        # Задаем объект класса SelectKBest для определения значимости признаков
        selector = feature_selection.SelectKBest(feature_selection.f_regression, k=features_count)
        # Обучаем объект
        selector.fit(X, y)
        
        # Составляем таблицу названиями признаков и их значимостью
        feature_scores_data = pd.DataFrame({
            'name': selector.get_feature_names_out(),
            'score': selector.scores_,
        })
        # Сортируем строки по значимости
        feature_scores_data = feature_scores_data.sort_values(by='score')
        
        # Выводим столбчатую диаграмму значимости признаков
        self.show_barplot_of_feature_scores(
            feature_scores_data['name'],
            feature_scores_data['score']
        )
        
        # Выводим список мультиколлинеарных признаков
        multicolinear_pairs = get_multicolinear_pairs(X, 0.7)
        print('Пары мультиколлинеарных признаков:')
        display(multicolinear_pairs)
