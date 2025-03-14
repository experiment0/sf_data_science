# Проект 7. Предсказание длительности поездки на такси (решение задачи регрессии)

## Оглавление

[Описание проекта](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Описание-проекта)\
[Какой кейс решаем?](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Какой-кейс-решаем)\
[Краткая информация о данных](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Краткая-информация-о-данных)\
[Этапы работы над проектом](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Этапы-работы-над-проектом)\
[Описание файлов проекта](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Описание-файлов-проекта)\
[Результат](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Результат)\
[Выводы](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Выводы)

## Описание проекта

В проекте реализовано решение задачи из соревнования на Kaggle [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview).\
По представленным данным требуется сделать предсказание длительности поездки на такси в Нью-Йорке.\
Срок соревнования закончился, данная работа является тренировочной.

## Какой кейс решаем?

По представленным данным о поездках на такси за 2016 год (около 1 500 000 записей) нужно построить модель,
которая будет предсказывать длительность поездки на такси.

## Краткая информация о данных

Данные исходной тренировочной и тестовой таблиц взяты из [соревнования](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview).\
Описание столбцов находится в файле с кодом проекта [taxi_ride_duration_prediction.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_07/taxi_ride_duration_prediction.ipynb)

Таблицы для расширения исходного набора данных были предолжены в готовом виде в обучающем курсе.\
Это облегчило задачу, ими нужно было просто воспользоваться.\
Это следующие таблицы.

- Таблица с данными о праздничных днях в 2016 году в США [holiday_data.zip](https://disk.yandex.ru/d/Zh1pZo_f3QQUtA)
- Данные о поездках, полученные с помощью сервиса [OSRM (Open Source Routing Machine)](https://en.wikipedia.org/wiki/Open_Source_Routing_Machine) [osrm_data_train.zip](https://disk.yandex.ru/d/_YOYziN_B-a4-Q)
- Таблица с данными о погоде в 2016 году в Нью-Йорке [weather_data.zip](https://disk.yandex.ru/d/wox1x1_5biSstA)

## Этапы работы над проектом

1. Проведен первичный анализ данных.
2. К исходному набору данных добавлены новые признаки \
(из перечисленных выше таблиц, а также с помощью преобразования имеющихся признаков).
3. Заполнены пропуски и удалены выбросы.
4. Проведен более подробный анализ нового набора данных.
5. Данные подготовлены для использования в модели:
    - удалены лишние признаки;
    - перекодированы категориальные признаки;
    - отобраны лучшие признаки;
    - оставшиеся признаки нормализованы.
6. Построено несколько моделей линейной регрессии и выбрана лучшая.
7. С помощью лучшей модели сделано предсказание для тестовой выборки из соревнования.

## Описание файлов проекта

Основной файл с кодом проекта [taxi_ride_duration_prediction.ipynb](https://github.com/experiment0/sf_data_science/blob/main/project_07/taxi_ride_duration_prediction.ipynb)

В папке `./classes` находятся следующие классы:
- [DataPreparation.py](https://github.com/experiment0/sf_data_science/blob/main/project_07/classes/DataPreparation.py) - содержит методы для обработки данных тренировочной и тестовой таблиц.\
Также хранит данные, актуальные для обеих таблиц.\
Создан для того, чтобы после обработки данных тренировочной таблицы было проще повторить те же манипуляции с тестовой таблицей.
- [Graphs.py](https://github.com/experiment0/sf_data_science/blob/main/project_07/classes/Graphs.py) - содержит методы для построения графиков.\
Позволяет сделать код построения графиков в проекте более коротким, а сами графики более единообразными по виду.

В папке `./helpers` содержатся файлы со вспомогательными функциями, которые было излишним делать методами классов.
- [geography.py](https://github.com/experiment0/sf_data_science/blob/main/project_07/helpers/geography.py) - содержит функции для расчета новых географических признаков
- [common.py](https://github.com/experiment0/sf_data_science/blob/main/project_07/helpers/common.py) - содержит вспомогательные функции общего назначения.

## Результат

Для дополненного набора данных построено несколько моделей линейной регрессии и выбрана лучшая.\
Файл с предсказанием для лучшей модели загружен на платформу с соревнованием.

## Выводы

Лучший результат показала модель [XGBoost](https://neerc.ifmo.ru/wiki/index.php?title=XGBoost).\
Значение метрики `RMSLE` (Root Mean Squared Log Error) 
- на тренировочной выборке **0.36807**
- на валидационной **0.38371**
- на тестовой выборке из соревнования **0.39351**

Время обучения составило меньше минуты при выбранных 500-х моделях.

:arrow_up:[к оглавлению](https://github.com/experiment0/sf_data_science/blob/main/project_07/README.md#Оглавление)
