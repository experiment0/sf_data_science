# Проект 6. Решение задачи классификации

## Оглавление

[Описание проекта](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Описание-проекта)\
[Какой кейс решаем?](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Какой-кейс-решаем)\
[Краткая информация о данных](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Краткая-информация-о-данных)\
[Этапы работы над проектом](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Этапы-работы-над-проектом)\
[Описание структуры проекта](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Описание-структуры-проекта)\
[Результат](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Результат)\
[Выводы](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Выводы)

## Описание проекта

В проекте реализовано решение задачи классификации.

## Какой кейс решаем?

По предоставленным данным о клиентах банка нужно построить модель, которая будет предсказывать, 
согласится клиент на педложение открыть депозит или нет.

## Краткая информация о данных

Данные предоставлены в рамках обучающего курса.\
Исходные данные можно скачать [здесь](https://disk.yandex.ru/d/agA97zeCmGdpuw).\
Описание столбцов исходной таблицы находится в файле [1_main.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/1_main.ipynb).

## Этапы работы над проектом

1. Проведен первичный анализ данных.
2. Сделана предварительная обработка данных (заполнение пропусков, удаление выбросов).
3. Проведено исследование обработанных данных \
на предмет распределения признаков в разрезе целевой переменной\
и корреляции признаков между собой.
3. Добавлены новые признаки, некоторые из которых оказались более сильными, чем исходные \
(дали бОльшую корреляцию с целевой переменной).
4. Построено несколько моделей классификации и выбраны лучшие.

## Описание структуры проекта

По ходу исследования данных стало понятно, что исходную таблицу с данными лучше разделить на две.\
В одну поместить новых клиентов (`data_newbies`), в другую старых (`data_participated`).\
Так как часть полей информативна только для старых клиентов, а для новых полезной информации в них нет.\
Как следствие, анализ и прогноз нужно сделать для каждой таблицы отдельно.\
Поэтому по причине большого объема работы для удобства чтения она разделена на несколько файлов.

### Файлы с описанием хода исследования

- [1_main.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/1_main.ipynb) - главный файл, с него следует начинать чтение.
- [2_1_prepare_data.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/2_1_prepare_data.ipynb) - описание первичной обработки данных.
    - [2_2_prepare_data_newbies.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/2_2_prepare_data_newbies.ipynb) - первичная обработка данных для таблицы `data_newbies`
    - [2_2_prepare_data_participated.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/2_2_prepare_data_participated.ipynb) - первичная обработка данных для таблицы `data_participated`
- [3_2_eda_newbies.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/3_2_eda_newbies.ipynb) - исследование данных таблицы `data_newbies`
- [3_2_eda_participated.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/3_2_eda_participated.ipynb) - исследование данных таблицы `data_participated`
- [4_2_feature_inginiring_newbies.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/4_2_feature_inginiring_newbies.ipynb) - отбор и преобразование признаков таблицы `data_newbies`
- [4_2_feature_inginiring_participated.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/4_2_feature_inginiring_participated.ipynb) - отбор и преобразование признаков таблицы `data_participated`
- [5_2_classification_newbies.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/5_2_classification_newbies.ipynb) - построение прогноза для данных из таблицы `data_newbies`
- [5_2_classification_participated.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_6/5_2_classification_participated.ipynb) - построение прогноза для данных из таблицы `data_participated`
           
### Вспомогательные функции

В папке `functions` содержаться файлы со вспомогательными функциями.
- [constants.py](https://github.com/experiment0/sf_data_science/blob/main/project_6/functions/constants.py) - содержит константы общего назначения, в том числе словарь с описанием данных для полей таблицы;
- [helpers.py](https://github.com/experiment0/sf_data_science/blob/main/project_6/functions/helpers.py) - содержит различные хелперы, также общего назначения;
- [graphics.py](https://github.com/experiment0/sf_data_science/blob/main/project_6/functions/graphics.py) - содержит функции для построения графиков;
- [display_content.py](https://github.com/experiment0/sf_data_science/blob/main/project_6/functions/display_content.py) - содержит функции для вывода однотипных частей контента (в том числе включающих графики);
- [prepare_data.py](https://github.com/experiment0/sf_data_science/blob/main/project_6/functions/prepare_data.py) - содержит вспомогательные функции для обработки данных.

### Классы для обработки данных

Обработка данных обеих таблиц местами имеет общее, а местами различается.\
Для более структурной записи общего и различий были созданы классы в папке `classes`.
- Класс `PrepareData` содержит общую для обеих таблиц обработку.
- А классы `PrepareDataNewbies` и `PrepareDataParticipated` наследуют общий класс `PrepareData`\
и содержат частные случаи обработки для таблиц `data_newbies` и  `data_participated` соответственно.

В файле [functions/prepare_data.py](https://github.com/experiment0/sf_data_science/blob/main/project_6/functions/prepare_data.py) при этом содержаться общие функции для обработки данных, которыми пожалуй не стоит загромождать классы.

### Класс для вывода навигации

Класс `Navigation` в папке `classes` используется для вывода навигации в файлах \
(хлебной крошки с заголовком вверху и ссылки на файл с родительским разделом внизу).

## Результат

**Для таблицы `data_newbies`** лучший результат показала модель `StackingClassifier`.\
Значение метрики $F_1$ на тестовой выборке равно `0.8392`.

**Для таблицы `data_participated`** лучший результат показала модель `RandomForestClassifier`.\
Значение метрики $F_1$ на тестовой выборке равно `0.8703`.

## Выводы

Разделение таблицы на две позволило получить более лучшие результаты.

Также замечено, что поиск оптимального значения гиперпараметров\
с помощью графика дает сравнимый, а иногда и более лучший результат, \
чем получение параметров с помощью `optuna`.

Чтобы получить больше пользы от `optuna`, нужно поэтапно корректировать область поиска параметров.

:arrow_up:[к оглавлению](https://github.com/experiment0/sf_data_science/blob/main/project_6/README.md#Оглавление)