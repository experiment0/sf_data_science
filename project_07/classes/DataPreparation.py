import numpy as np
import pandas as pd
from enum import Enum
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import cluster
from typing import Tuple

# Вспомогательные функции
from helpers.geography import get_haversine_distance, get_angle_direction


# Имена полей таблицы
class F(Enum):
    # ИСХОДНЫЕ ПРИЗНАКИ
    # уникальный идентификатор поездки
    ID = 'id'
    # уникальный идентификатор поставщика услуг (таксопарка), связанного с записью поездки
    VENDOR_ID = 'vendor_id'
    # дата и время, когда был включён счётчик поездки
    PICKUP_DATETIME = 'pickup_datetime'
    # дата и время, когда счётчик был отключён
    DROPOFF_DATETIME = 'dropoff_datetime'
    # долгота, на которой был включён счётчик
    PICKUP_LONGITUDE = 'pickup_longitude'
    # широта, на которой был включён счётчик
    PICKUP_LATITUDE = 'pickup_latitude'
    # долгота, на которой счётчик был отключён
    DROPOFF_LONGITUDE = 'dropoff_longitude'
    # широта, на которой счётчик был отключён
    DROPOFF_LATITUDE = 'dropoff_latitude'
    # количество пассажиров в транспортном средстве (введённое водителем значение)
    PASSENGER_COUNT = 'passenger_count'
    # флаг, который указывает, сохранилась ли запись о поездке
    # в памяти транспортного средства перед отправкой поставщику
    # (Y — хранить и пересылать, N — не хранить и не пересылать поездку)
    STORE_AND_FWD_FLAG = 'store_and_fwd_flag'
    # продолжительность поездки в секундах
    TRIP_DURATION = 'trip_duration'
    
    # НОВЫЕ ПРИЗНАКИ
    # дата включения счётчика/начала поездки (без времени)
    PICKUP_DATE = 'pickup_date'
    # час включения счётчика
    PICKUP_HOUR = 'pickup_hour'
    # день недели, в который был включён счётчик (0 - понедельник)
    PICKUP_DAY_OF_WEEK = 'pickup_day_of_week'
    # начата ли поездка в праздничный день (1 — да, 0 — нет)
    IS_PICKUP_IN_HOLIDAY = 'is_pickup_in_holiday'
    # кратчайшее дорожное расстояние (в метрах) из точки, в которой был включён счётчик, 
    # до точки, в которой счётчик был выключен (согласно OSRM)
    TOTAL_DISTANCE = 'total_distance'
    # наименьшее время поездки (в секундах) из точки, в которой был включён счётчик, 
    # до точки, в которой счётчик был выключен (согласно OSRM)
    TOTAL_TRAVEL_TIME = 'total_travel_time'
    # количество дискретных шагов, которые должен выполнить водитель 
    # (поворот налево/поворот направо/ехать прямо и т. д., согласно OSRM)
    NUMBER_OF_STEPS = 'number_of_steps'
    # расстояние по формуле Гаверсинуса между точкой, в которой был включён счетчик, 
    # и точкой, в которой счётчик был выключен
    HAVERSINE_DISTANCE  = 'haversine_distance'
    # направление движения из точки, в которой был включён счётчик, 
    # в точку, в которой счётчик был выключен
    ANGLE_DIRECTION = 'angle_direction'
    # географический кластер, к которому относится поездка 
    # (определенный с помощью KMeans на тренировочном наборе данных)
    GEO_CLUSTER = 'geo_cluster'
    # температура
    TEMPERATURE = 'temperature'
    # видимость
    VISIBILITY = 'visibility'
    # средняя скорость ветра
    WIND_SPEED = 'wind speed'
    # количество осадков
    PRECIP = 'precip'
    # погодные явления (Rain — дождь; Snow — снег; Fog — туман, None — отсутствие погодных явлений)
    EVENTS = 'events'
    # целевая переменная TRIP_DURATION, переведенная в логарифмическую шкалу 
    TRIP_DURATION_LOG = 'trip_duration_log'
    

class DataPreparation:
    """Класс, для преобразования данных тренировочной и тестовой таблиц.
    """
    def __init__(self) -> None:
        pass
    

    def set_holiday_data(self, holiday_data: pd.DataFrame):
        """Устанавливает поле с данными таблицы с праздничными датами
        (таблица нужна для создания новых признаков)

        Args:
            holiday_data (pd.DataFrame): таблица с праздничными датами
        """
        self.holiday_data = holiday_data.copy()
    
    
    def set_osrm_data(self, osrm_data: pd.DataFrame):
        """Устанавливает поле с данными таблицы, полученной с помощью сервиса
        OSRM (Open Source Routing Machine)

        Args:
            osrm_data (pd.DataFrame): таблица с данными от сервиса OSRM
        """
        self.osrm_data = osrm_data.copy()
    
    
    def set_weather_data(self, weather_data: pd.DataFrame):
        """Устанавливает поле с таблицей с данными о погоде

        Args:
            weather_data (pd.DataFrame): таблица с данными о погоде
        """
        self.weather_data = weather_data.copy()
        
    
    def add_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавляет признаки о дате и времени поездки

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        datetime_format = '%Y-%m-%d %H:%M:%S'
        
        # Переводим даты из строки в формат datetime
        data[F.PICKUP_DATETIME.value] = \
            pd.to_datetime(data[F.PICKUP_DATETIME.value], format=datetime_format)
            
        if (F.DROPOFF_DATETIME.value in data.columns):
            data[F.DROPOFF_DATETIME.value] = \
                pd.to_datetime(data[F.DROPOFF_DATETIME.value], format=datetime_format)
        
        # Дата начала поездки
        data[F.PICKUP_DATE.value] = data[F.PICKUP_DATETIME.value].dt.date
        # Час начала поездки
        data[F.PICKUP_HOUR.value] = data[F.PICKUP_DATETIME.value].dt.hour
        # День недели начала поездки
        data[F.PICKUP_DAY_OF_WEEK.value] = data[F.PICKUP_DATETIME.value].dt.day_of_week
        
        return data
    
    
    def add_holiday_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавляет в таблицу бинарный признак, являлся ли день поездки праздничным днем

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # Список с датами праздников в года, в которые совершались поездки
        holiday_dates = list(self.holiday_data['date'])
        
        # Если дата поездки попадает на праздник, выставляем 1, иначе 0
        data[F.IS_PICKUP_IN_HOLIDAY.value] = \
            data[F.PICKUP_DATE.value].apply(lambda value: 1 if str(value) in holiday_dates else 0)
        
        return data
    
    
    def add_osrm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавляем признаки из таблицы, полученной с помощью сервиса
        OSRM (Open Source Routing Machine)

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # Отбираем только столбцы, которые будем присоединять к исходной таблице
        osrm_data_necessary = self.osrm_data.loc[:, ['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
        
        # Присоединяем столбцы к исходной таблице
        data = data.merge(osrm_data_necessary, on=F.ID.value, how='left')
        
        return data
    
    
    def add_geographical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавляет новые географические признаки

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # Считаем расстояние между двумя точками по формуле гаверсинуса (в километрах)
        data[F.HAVERSINE_DISTANCE.value] = \
            data.apply(lambda data: get_haversine_distance(
                data[F.PICKUP_LATITUDE.value],
                data[F.PICKUP_LONGITUDE.value],
                data[F.DROPOFF_LATITUDE.value],
                data[F.DROPOFF_LONGITUDE.value],
            ), axis=1)
        
        # Считаем угол направления движения (в градусах)
        data[F.ANGLE_DIRECTION.value] = \
            data.apply(lambda data: get_angle_direction(
                data[F.PICKUP_LATITUDE.value],
                data[F.PICKUP_LONGITUDE.value],
                data[F.DROPOFF_LATITUDE.value],
                data[F.DROPOFF_LONGITUDE.value],
            ), axis=1) 
        
        return data
    
    
    def set_geo_cluster_kmeans(self, data: pd.DataFrame):
        """Обучает объект класса KMeans для кластеризации точек с координатами начала и конца поездки.
        Сохраняет обученный объект в поле.
        В дальнейшем объект будет использован для создания признака с классом в тренировочной и тестовой таблицах.

        Args:
            data (pd.DataFrame): тренировочная таблица с данными
        """
        # Создаём обучающую выборку из географических координат всех точек
        coords = np.hstack((data[[F.PICKUP_LATITUDE.value, F.PICKUP_LONGITUDE.value]],
                            data[[F.DROPOFF_LATITUDE.value, F.DROPOFF_LONGITUDE.value]]))
        # Устанавливаем параметры кластеризации
        kmeans = cluster.KMeans(n_clusters=10, random_state=42)
        # Обучаем алгоритм
        kmeans.fit(coords)
        # Сохраняем обученный объект в поле
        self.geo_cluster_kmeans = kmeans
    
    
    def add_geo_cluster_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавляет признак с номером кластера поездки, определенным с помощью класса KMeans

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # Координаты точек начала и конца поездок
        coords = np.hstack((data[[F.PICKUP_LATITUDE.value, F.PICKUP_LONGITUDE.value]],
                            data[[F.DROPOFF_LATITUDE.value, F.DROPOFF_LONGITUDE.value]]))
        
        # Предсказываем кластеры
        predictions = self.geo_cluster_kmeans.predict(coords)
        
        # Создаем новый столбец со значениями кластеров
        data[F.GEO_CLUSTER.value] = predictions
        
        return data
    
    
    def add_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавляет признаки с данными о погоде в день и час поездки

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # Отбираем из таблицы с данными о погоде только те признаки, которые будем объединять с нашими данными
        weather_data_necessary = \
            self.weather_data.loc[:, ['temperature', 'visibility', 'wind speed', 'precip', 'events', 'date', 'hour']]
        
        # В обеих таблицах приводим столбец с датой к строковому типу
        data[F.PICKUP_DATE.value] = data[F.PICKUP_DATE.value].astype(str)
        weather_data_necessary['date'] = weather_data_necessary['date'].astype(str)
        
        # Объединяем выбранные колонки с нашей таблицей
        data = data.merge(
            weather_data_necessary,
            how='left',
            left_on=[F.PICKUP_DATE.value, F.PICKUP_HOUR.value],
            right_on=['date', 'hour']
        )
        
        # Удаляем столбцы с датой и часом поездки, т.к. они дублируют уже имеющиеся
        data.drop(columns=['date', 'hour'], inplace=True)
        
        return data
    
    
    def set_weather_median_data(self):
        """Формирует и запоминает таблицу, составленную из медианных значений данных о погоде по датам.
        Таблица будет нужна в дальнейшем для заполнения пропусков.
        """
        # Составим таблицу из медианных значений признаков для каждой даты
        weather_median_data = self.weather_data\
            .groupby('date')[['temperature', 'visibility', 'wind speed', 'precip']].agg('median')
        
        # Запомним полученную таблицу
        self.weather_median_data = weather_median_data
    
    
    def set_fill_null_dict(self, data: pd.DataFrame):
        """Определяет и запоминает значения для заполнения пропусков.
        Для дальнейшего использования в тренировочной и тестовой таблицах.

        Args:
            data (pd.DataFrame): тренировочная таблица с данными
        """
        fill_null_dict = {}
        
        # Для пропусков в колонке погодных явлений будем указывать их отсутствие
        fill_null_dict[F.EVENTS.value] = 'None'
        
        # Список имен признаков, полученных с помощью сервиса OSRM
        osrm_fields_list = [
            F.TOTAL_DISTANCE.value,
            F.TOTAL_TRAVEL_TIME.value,
            F.NUMBER_OF_STEPS.value,
        ]
        for field in osrm_fields_list:
            # Пропуски будем заполнять медианным значением
            fill_null_dict[field] = data[field].median()
        
        # Запомним полученный словарь
        self.fill_null_dict = fill_null_dict
        
    
    def fill_null_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Заполняет пропуски в данных

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # Временно добавим в таблицу медианные значения погодных признаков по датам
        data = data.merge(
            self.weather_median_data,
            how='left',
            left_on=[F.PICKUP_DATE.value],
            right_on=['date'],
            suffixes=(None, '_median'),
        )
        
        # Список признаков с данными о погоде, для которых будем заполнять пропуски
        weather_fields_list = [
            F.TEMPERATURE.value,
            F.VISIBILITY.value,
            F.WIND_SPEED.value,
            F.PRECIP.value,
        ]
        
        # Для каждого признака заполним пропуск медианным значением для текущей даты
        for field in weather_fields_list:
            data[field] = data[field].fillna(data[f'{field}_median'])
            
        # Удалим временные колонки с медианными значениями
        columns_to_drop = list(map(lambda value: f'{value}_median', weather_fields_list))
        data.drop(columns=columns_to_drop, inplace=True)
        
        # Идем по запомненному ранее словарю со значениями для заполнения пропусков
        for field, value in self.fill_null_dict.items():
            # И заполняем пропуски
            data[field] = data[field].fillna(value)
        
        return data
    
    
    def drop_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Удаляет выбросы в данных

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # Удалим поездки, длительность которых превышает 24 часа
        trip_duration_mask = data[F.TRIP_DURATION.value] > 24*60*60
        data.drop(data[trip_duration_mask].index, inplace=True)
        
        # Удалим поездки, средняя скорость которых превышает 300 км/ч
        def is_speed_exceeded(data):
            avg_speed = (data[F.TOTAL_DISTANCE.value] / data[F.TRIP_DURATION.value]) * 3.6
            return avg_speed > 300

        indexes_with_speed_outliers = data[data.apply(is_speed_exceeded, axis='columns')].index
        data.drop(indexes_with_speed_outliers, inplace=True)
        
        return data
    
    
    def add_target_log_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавляет колонку с целевым признаком, переведенным в логарифмическую шкалу

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        data[F.TRIP_DURATION_LOG.value] = np.log(data[F.TRIP_DURATION.value] + 1)
        
        return data
    
    
    def get_source_target_feature(self, target_log_value: float) -> float:
        """Возвращает исходное значение из значения целевой переменной, 
        переведенной в логарифмический масштаб

        Args:
            target_log_value (float): значение в логарифмическом масштабе

        Returns:
            float: исходное значение (которое было до перевода в логарифмический масштаб)
        """
        return np.exp(target_log_value) - 1
    
    
    def drop_unnecessary_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Удаляет лишние признаки

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: таблица с удаленными признаками (тренировочная или тестовая)
        """
        columns_to_drop = [
            F.ID.value,
            F.PICKUP_DATETIME.value,
            F.PICKUP_DATE.value,
        ]
        
        # В тестовой таблице нет этих колонок, поэтому уточняем наличие
        if (F.DROPOFF_DATETIME.value in data.columns):
            columns_to_drop.append(F.DROPOFF_DATETIME.value)
        if (F.TRIP_DURATION.value in data.columns):
            columns_to_drop.append(F.TRIP_DURATION.value)
            
        data.drop(columns=columns_to_drop, inplace=True)
        
        return data
    
    
    def encoding_binary_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Кодирует бинарные признаки

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: таблица с перекодированными бинарными признаками (тренировочная или тестовая)
        """
        data[F.VENDOR_ID.value] = data[F.VENDOR_ID.value].apply(
            lambda value: 0 if value == 1 else 1
        )
        data[F.STORE_AND_FWD_FLAG.value] = data[F.STORE_AND_FWD_FLAG.value].apply(
            lambda value: 0 if value == 'N' else 1
        )
        
        return data
    
    
    def set_one_hot_encoder(self, data: pd.DataFrame):
        """Обучает и запоминает объект класса OneHotEncoder для кодирования категориальных признаков

        Args:
            data (pd.DataFrame): исходная тренировочная таблица
        """
        # Признаки, которые собираемся перекодировать
        columns_to_encode = [F.PICKUP_DAY_OF_WEEK.value, F.GEO_CLUSTER.value, F.EVENTS.value]
        
        # объект класса OneHotEncoder
        one_hot_encoder = preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore')
        
        # обучаем енкодер
        one_hot_encoder.fit(data[columns_to_encode])
        
        # запоминаем колонки, которые будем кодировать
        self.columns_to_one_hot_encode = columns_to_encode
        # запоминаем обученный объект
        self.one_hot_encoder = one_hot_encoder
        
    
    def encoded_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Перекодирует категориальные признаки

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: таблица с перекодированными признаками (тренировочная или тестовая)
        """
        # перекодированные данные в виде массива
        encoded_data_array = self.one_hot_encoder.transform(
            data[self.columns_to_one_hot_encode]
        ).toarray() 

        # имена перекодированных колонок
        encoded_column_names = self.one_hot_encoder.get_feature_names_out(self.columns_to_one_hot_encode)

        # собираем таблицу из данных и колонок
        data_onehot = pd.DataFrame(
            encoded_data_array,
            columns=encoded_column_names,
        )
        
        # добавим полученную таблицу с закодированными признаками
        data = pd.concat(
            [data.reset_index(drop=True).drop(self.columns_to_one_hot_encode, axis=1), data_onehot], 
            axis=1
        )
        
        return data
    
    
    def prepare_data(self, data: pd.DataFrame, is_data_train: bool = True) -> pd.DataFrame:
        """Готовит данные для построения модели

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)
            is_data_train (bool, optional): Флаг, являются ли данные тренировочными. По умолчанию True.

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочная или тестовая)
        """
        # добавляем признаки о дате и времени поездки
        data = self.add_datetime_features(data)
        # добавляем признак, является ли день поездки праздничным
        data = self.add_holiday_feature(data)
        # добавляем признаки, полученные с помощью сервиса OSRM (Open Source Routing Machine)
        data = self.add_osrm_features(data)
        # добавляем географические признаки
        data = self.add_geographical_features(data)

        if (is_data_train):
            # обучаем и запоминаем объект класса KMeans для кластеризации поездок
            self.set_geo_cluster_kmeans(data)

        # добавляем поле с кластером поездки
        data = self.add_geo_cluster_feature(data)
        # добавляем признаки с данными о погоде в день поездки
        data = self.add_weather_features(data)

        if (is_data_train):
            # сформируем и запомним таблицу с медианными значениями погодных данных по датам
            self.set_weather_median_data()
            # определяем и запоминаем данные для заполнения пропусков
            self.set_fill_null_dict(data)

        # заполняем пропуски
        data = self.fill_null_values(data)        
        
        if (is_data_train):
            # удаляем выбросы
            data = self.drop_outliers(data)
            # добавим колонку с целевым признаком, переведенным в логарифмическую шкалу
            data = self.add_target_log_feature(data)
        
        # удалим лишние признаки
        data = self.drop_unnecessary_features(data)
        
        # перекодируем бинарные признаки
        data = self.encoding_binary_features(data)
        
        if (is_data_train):
            # обучим и запомним объект класса OneHotEncoder для кодирования категориальных признаков
            self.set_one_hot_encoder(data)
        
        # перекодируем категориальные признаки
        data = self.encoded_categorical_features(data)
        
        return data
    
    
    def get_predictors_and_target(self, data: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.Series
    ]:
        """Формирует таблицу с предикторами и столбец с целевой переменной в логарифмическом масштабе

        Args:
            data (pd.DataFrame): данные тренировочной таблицы

        Returns:
            Tuple[ pd.DataFrame, pd.Series ]: 
            - таблица с предикторами
            - столбец с целевой переменной в логарифмическом масштабе
        """
        # Удаляем из предикторов колонку с целевой переменной, 
        # переведенной в логарифмический масштаб
        X = data.drop(columns=[F.TRIP_DURATION_LOG.value])
        
        # Предсказывать будем целевую переменную в логарифмическом масштабе
        y_log = data[F.TRIP_DURATION_LOG.value]
        
        return X, y_log
    
    
    def get_train_and_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    ]:
        """Формирует тренировочную и валидационную выборки

        Args:
            X (pd.DataFrame): таблица с предикторами
            y (pd.Series): столбец со значениями целевой переменной

        Returns:
            Tuple[ pd.DataFrame, pd.DataFrame, pd.Series, pd.Series ]: 
                - предикторы тренировочной таблицы
                - предикторы валидационной таблицы
                - целевая переменная тренировочной таблицы
                - целевая переменная валидационной таблицы
        """
        # Делим выборку
        X_train, X_valid, y_train, y_valid = \
            model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
        
        return X_train, X_valid, y_train, y_valid
    
    
    def set_best_feature_names(self, best_feature_names: np.ndarray):
        """Запоминает набор лучших предикторов

        Args:
            best_feature_names (np.ndarray): набор лучших предикторов
        """
        self.best_feature_names = best_feature_names
        
    
    def get_best_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Возвращает колонки только с лучшими предикторами

        Args:
            data (pd.DataFrame): исходная таблица (тренировочная или тестовая)

        Returns:
            pd.DataFrame: таблица, в которой остались только колонки с лучшими предикторами
        """
        return data.loc[:, self.best_feature_names]
    
    
    def set_min_max_scaler(self, data: pd.DataFrame):
        """Обучает и запоминает объект класса MinMaxScaler для масштабирования данных

        Args:
            data (pd.DataFrame): таблица с тренировочными данными
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(data)
        
        self.min_max_scaler = min_max_scaler
        
    
    def get_scaled_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Масштабирует данные таблицы

        Args:
            data (pd.DataFrame): исходная таблица с данными (тренировочная или тестовая)

        Returns:
            pd.DataFrame: таблица с масштабированными данными (тренировочная или тестовая)
        """
        scaled_data = self.min_max_scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=self.best_feature_names)
        
        return scaled_data
    
    
    def get_best_scaled_features(
        self, 
        predictors_data: pd.DataFrame, 
        is_data_train: bool = True
    ) -> pd.DataFrame:
        """Выделяет лучшие для предсказания признаки признаки и масштабирует их

        Args:
            predictors_data (pd.DataFrame): таблица с предикторами (тренировочная или тестовая)
            is_data_train (bool, optional): Флаг, являются ли данные тренировочными. По умолчанию True.

        Returns:
            pd.DataFrame: преобразованная таблица с данными (тренировочными или тестовыми)
        """
        # выделим лучшие предикторы в тренировочной и валидационной выборках
        predictors_data = self.get_best_features(predictors_data)

        if (is_data_train):
            # обучим и запомним объект для масштабирования данных
            self.set_min_max_scaler(predictors_data)

        # получим масштабированные данные
        predictors_data = self.get_scaled_data(predictors_data)
        
        return predictors_data