# Вспомогательные функции для обработки данных

import random
from enum import Enum
from typing import Tuple
import numpy as np
import re
import pandas as pd

from helpers.constants import RANDOM_STATE


# Имена столбцов таблицы (Field -> F)
class F(Enum):
    # номер счёта-фактуры
    INVOICE_NO = 'InvoiceNo'
    # код товара
    STOCK_CODE = 'StockCode'
    # название товара
    DESCRIPTION = 'Description'
    # количество каждого товара за транзакцию
    QUANTITY = 'Quantity'
    # количество возвращенных впоследствии товаров для данной транзакции
    QUANTITY_CANCELED = 'QuantityCanceled'
    # дата и время выставления счёта/проведения транзакции
    INVOICE_DATE = 'InvoiceDate'
    # месяц выставления счёта/проведения транзакции
    INVOICE_MONTH = 'InvoiceMonth'
    # число месяца, в которое был выставлен счет/проведена транзакция
    INVOICE_MONTH_DAY = 'InvoiceMonthDay'
    # номер дня недели, в который был выставлен счет/проведена транзакция
    INVOICE_WEEK_DAY = 'InvoiceWeekDay'
    # час выставления счёта/проведения транзакции
    INVOICE_HOUR = 'InvoiceHour'
    # цена за единицу товара в фунтах стерлингов
    UNIT_PRICE = 'UnitPrice'
    # итоговая сумма покупки
    TOTAL_PRICE = 'TotalPrice'
    # идентификатор клиента
    CUSTOMER_ID = 'CustomerID'
    # название страны, в которой проживает клиент
    COUNTRY = 'Country'
    # количество дней, которое прошло с момента последней покупки клиента
    RECENCY = 'Recency'
    # общее количество уникальных заказов клиента
    FREQUENCY = 'Frequency'
    # сколько всего денег потратил клиент
    MONETARY = 'Monetary'
    # метки классов, определенные с помощью алгоритма k-means
    LABEL_KMEANS = 'label_kmeans'
    # метки, предсказанные на отложенной выборке с помощью итоговой модели
    LABEL_PREDICTED = 'label_predicted'


# Имена временных столбцов таблицы (Field temp -> FT)
class FT(Enum):
    # буквенный префикс счета-фактуры
    INVOICE_NO_PREFIX = 'InvoiceNoPrefix'
    # буквенное окончание счета-фактуры
    INVOICE_NO_POSTFIX = 'InvoiceNoPostfix'
    # буквенный префикс кода товара
    STOCK_CODE_PREFIX = 'StockCodePrefix'
    # буквенное окончание кода товара
    STOCK_CODE_POSTFIX = 'StockCodePostfix'
    
    
def get_splited_data(
    source_data: pd.DataFrame, 
    sample_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет исходну выборку на отложенную и тренировочную

    Args:
        data_source (pd.DataFrame): исходная таблица
        sample_size (float, optional): доля уникальных пользователей, 
            которых нужно выделить в отложенную выборку. 
            По умолчанию 0.1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: тренировочная и отложенная выборки
    """
    # зафиксируем начальное число для воспроизводимости генерации случайной выборки
    random.seed(RANDOM_STATE)    
    # количество уникальных пользователей
    customers_count = source_data[F.CUSTOMER_ID.value].nunique()    
    # количество пользователей для отложенной выборки
    samples_count = round(customers_count * sample_size)    
    # список всех уникальных идентификаторов пользователей
    customer_ids = list(source_data[F.CUSTOMER_ID.value].unique())
    # список уникальных идентификаторов пользователей для отложенной выборки
    sample_customer_ids = random.sample(customer_ids, samples_count)
    # формируем маску для отбора только тех идентификаторов пользователей, 
    # которые входят в отложенную выборку
    sample_mask = source_data[F.CUSTOMER_ID.value].isin(sample_customer_ids)
    # формируем данные отложенной выборки 
    # (удаляем из исходной таблицы все записи, не подходящие под маску)
    sample_data = source_data.drop(source_data[~sample_mask].index) 
    # формируем данные тренировочной выборки 
    # (удаляем из исходной таблицы все записи, подходящие под маску)
    train_data = source_data.drop(source_data[sample_mask].index)
    
    return train_data, sample_data
    

def get_str_prefix(value: str) -> str:
    """Возвращает префикс строки, который состоит из символов

    Args:
        value (str): изначальная строка

    Returns:
        str: префикс из символов
    """
    match = re.match('\D*', value)
    return match.group()


def get_str_postfix(value: str) -> str:
    """Возвращает окончание из символов в строке

    Args:
        value (str): изначальная строка

    Returns:
        str: окончание, состоящее из символов
    """
    match = re.findall('\D*\Z', value)
    return match[0]


def get_quantity_canceled(data: pd.DataFrame) -> pd.Series:
    """Определяет и возвращает столбец с количеством возвращенных товаров для каждой транзакции

    Args:
        data (pd.DataFrame): исходная таблица

    Returns:
        pd.Series: столбец с количеством возвращенных товаров для каждой транзакции
    """
    # Инициализируем нулями Series той же длины, что и столбцы таблицы
    quantity_canceled_data = pd.Series(np.zeros(data.shape[0]), index=data.index)
    # Выделяем в отдельную таблицу транзакции с отрицательным количеством товаров
    negative_quantity_data = data[(data[F.QUANTITY.value] < 0)].copy()
    
    # Обходим все транзакции-возвраты
    for index_neg, row_neg in negative_quantity_data.iterrows():
        # Создаём DataFrame из всех транзакций, противоположных возвратам
        positive_quantity_data = data[(data[F.CUSTOMER_ID.value] == row_neg[F.CUSTOMER_ID.value]) &
                       (data[F.STOCK_CODE.value]  == row_neg[F.STOCK_CODE.value]) & 
                       (data[F.INVOICE_DATE.value] < row_neg[F.INVOICE_DATE.value]) & 
                       (data[F.QUANTITY.value] > 0) &
                       (data[F.QUANTITY.value] >= -row_neg[F.QUANTITY.value])].copy()
        # Если транзация-возврат не имеет противоположной — ничего не делаем
        if (positive_quantity_data.shape[0] == 0): 
            continue
        # Если транзакция-возврат имеет только одну противоположную транзакцию
        # Добавляем количество возвращённого товара в столбец QuantityCanceled 
        elif (positive_quantity_data.shape[0] == 1): 
            order_index = positive_quantity_data.index[0]
            quantity_canceled_data.loc[order_index] = -row_neg[F.QUANTITY.value]       
        # Транзакция-возврат имеет несколько противоположных транзакций
        # Вносим количество возвращённого товара в столбец QuantityCanceled для той транзакции на покупку,
        # в которой количество товара > (-1) * (количество товаров в транзакции-возврате)
        elif (positive_quantity_data.shape[0] > 1): 
            positive_quantity_data.sort_index(axis=0, ascending=False, inplace=True)        
            for index_pos, row_pos in positive_quantity_data.iterrows():
                if row_pos[F.QUANTITY.value] < -row_neg[F.QUANTITY.value]: 
                    continue
                quantity_canceled_data.loc[index_pos] = -row_neg[F.QUANTITY.value]
                break    
    return quantity_canceled_data
    

def get_total_price(unit_price: float, quantity: int, quantity_canceled: int) -> float:
    """Считает итоговую сумму покупки

    Args:
        unit_price (float): цена за единицу товара
        quantity (int): количество купленных изначально единиц товара
        quantity_canceled (int): количество возвращенных единиц товара

    Returns:
        float: итоговая сумма покупки
    """
    return unit_price * (quantity - quantity_canceled)


def get_prepared_data(data: pd.DataFrame) -> pd.DataFrame:
    """Возвращает обработанные данные

    Args:
        data (pd.DataFrame): исходная таблица

    Returns:
        pd.DataFrame: таблица с обработанными данными
    """
    data = data.copy()
    
    # Переведем признак с датой в формат datetime
    data[F.INVOICE_DATE.value] = pd.to_datetime(data[F.INVOICE_DATE.value])
    
    # Удалим пропуски
    data.dropna(subset=[F.CUSTOMER_ID.value], inplace=True)
    
    # Удалим дубликаты
    data.drop_duplicates(ignore_index=True, inplace=True)
    
    # Удалим транзакции, которые не являются оплатой товаров (почтовые расходы и прочее)
    stock_code_mask = data[F.STOCK_CODE.value].isin(['POST', 'M', 'D', 'DOT', 'CRUK', 'BANK CHARGES', 'C2'])
    data.drop(data[stock_code_mask].index, inplace=True)
    
    # Определим количество возвращенных товаров для каждой транзакции
    data[F.QUANTITY_CANCELED.value] = get_quantity_canceled(data)
    data[F.QUANTITY_CANCELED.value] = data[F.QUANTITY_CANCELED.value].astype(int)
    
    # Удалим строки с отрицательным количеством товаров
    negative_quantity_mask = data[F.QUANTITY.value] < 0
    data.drop(data[negative_quantity_mask].index, inplace=True)
    
    # Посчитаем итоговую сумму покупок
    data[F.TOTAL_PRICE.value] = data.apply(
        lambda row: get_total_price(
            row[F.UNIT_PRICE.value], 
            row[F.QUANTITY.value], 
            row[F.QUANTITY_CANCELED.value]
        ), axis=1
    )
    
    return data


def get_rfm_data(
    data: pd.DataFrame, 
    start_recency_date: pd.Timestamp
) -> pd.DataFrame:
    """Формирует таблицу с RFM признаками

    Args:
        data (pd.DataFrame): исходная таблица с данными
        start_recency_date (pd.Timestamp): дата, от которой отсчитывается 
        количество дней до последней покупки (признак Requency)

    Returns:
        pd.DataFrame: таблица с RFM признаками
    """
    # Рассчитаем для каждого заказа разницу с днем отсчета
    data[F.RECENCY.value] = data[F.INVOICE_DATE.value].rsub(start_recency_date).dt.days
    
    # Соберем данные для каждого клиента
    rfm_data = data.groupby(F.CUSTOMER_ID.value).agg(
        Recency=(F.RECENCY.value, 'min'), # сколько дней назад в последний раз делал заказ
        Frequency=(F.INVOICE_NO.value, 'nunique'), # общее количество уникальных заказов
        Monetary=(F.TOTAL_PRICE.value, 'sum'), # сколько всего потратил денег
    )

    return rfm_data