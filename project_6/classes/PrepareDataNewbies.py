from datetime import datetime
from pandas import DataFrame

from classes.PrepareData import PrepareData, new_features
from functions.constants import (
    CURRENT_YEAR
)

# Столбцы для удаления из таблицы data_newbies
columns_to_drop = ['duration_scale', 'age_scale', 'housing', 'loan', 'default', 'season',
                   'job', 'month', 'day', 'age']

# Количество признаков, которые оставим в таблице data_newbies
best_features_count = 21


class PrepareDataNewbies(PrepareData):
    def __init__(
        self, 
        data_source: DataFrame, 
        new_features: list = new_features, 
        columns_to_drop: list = columns_to_drop,
        best_features_count: int = best_features_count,
    ) -> None:
        """Выполняет специфические преобразования для таблицы data_newbies

        Args:
            data_source (DataFrame): исходная таблица с данными
            new_features (list, optional): список новых признаков. По умолчанию new_features.
            columns_to_drop (list, optional): список признаков для удаления перед кодированием. 
                                              По умолчанию columns_to_drop.
            best_features_count (int, optional): Количество лучших признаков, которые оставляем. 
                                                 По умолчанию best_features_count.
        """
        super().__init__(data_source, new_features, columns_to_drop, best_features_count)
        
        
    def set_specific_data(self):
        """Формирует таблицу с клиентами, не участвовавшими ранее в кампаниях банка.
        Преобразования определены по результатам исследования.
        """
        self.primary_prepare_data()
        
        # В признаке `previous` содержится количество контактов до текущей кампании.
        # Если оно равно 0, значит текущая кампания для клиента первая.
        mask_not_newbies = self.data['previous'] > 0
        
        # Удаляем строки по маске
        self.data.drop(index=self.data[mask_not_newbies].index, axis=0, inplace=True)
        
        # Удаляем столбцы, которые не несут полезной информации для клиентов, 
        # не участвовавших раньше в кампаниях банка
        self.data.drop(columns=['pdays', 'previous', 'poutcome'], inplace=True)
        
        
    def specific_prepare_data(self):
        """Выполняет предварительное преобразование данных для таблицы data_newbies.
        Преобразования определены по результатам исследования.
        """
        # Фильтруем данные, специфические для данной таблицы
        self.set_specific_data()
        
        # Делаем общие преобразования
        self.common_prepare_data()
        
    
    def get_ringing_type(self, date: datetime) -> str:
        """Возвращает тип прозвона в зависимости от даты

        Args:
            date (datetime): календарная дата

        Returns:
            str: предполагаемый тип прозвона
        """
        def get_date(day, month):
            date = datetime(CURRENT_YEAR, month, day)
            return date 

        # 4 мая - 30 мая
        if (date >= get_date(4, 5) and date <= get_date(30, 5)):
            # Прозвон по кредиту на жилье
            return 'housing'
        # 30 апреля
        if (date == get_date(30, 4)):
            # Прозвон по депозиту
            return 'deposit'
        if (
            # 28 января - 6 февраля
            (date >= get_date(28, 1) and date <= get_date(6, 2)) or
            # 14 апреля - 20 июня
            (date >= get_date(14, 4) and date <= get_date(20, 6)) or
            # 7 июля - 29 августа
            (date >= get_date(7, 7) and date <= get_date(29, 8)) or
            # 17 ноября - 21 ноября
            (date >= get_date(17, 11) and date <= get_date(21, 11))
        ):
            # Общий прозвон
            return 'common'
        # Активного прозвона нет
        return 'weak'
    
    
    def get_job_loyal(self, job: str) -> str:
        """Возвращает степень готовности открыть депозит в зависимости от сферы занятости

        Args:
            job (str): сфера занятости клиента

        Returns:
            str: степень готовности открыть депозит
        """
        if (job in ['blue-collar', 'entrepreneur', 'housemaid', 'services']):
            return 1
        if (job in ['self-employed', 'admin.', 'technician', 'management']):
            return 2
        if (job == 'unemployed'):
            return 3
        if (job in ['retired', 'student']):
            return 4
        
        
    def add_new_specific_features(self):
        """Добавляет новые признаки из списка self.new_features, 
        специфически сформированные для таблицы data_newbies
        """
        # Выполняем предварительное преобразование данных
        self.specific_prepare_data()
        
        # Формируем новые признаки из списка self.new_features , общие для обеих таблиц
        self.add_new_common_features()
        
        # Формируем новые признаки, специфические для данной таблицы
        if ('ringing_type' in self.new_features):
            # Тип прозвона
             self.data['ringing_type'] =  self.data['contact_date'].apply( self.get_ringing_type)
        
        if ('job_loyal' in self.new_features):
            # Уровень готовности открыть депозит в зависимости от сферы занятости
             self.data['job_loyal'] =  self.data['job'].apply(self.get_job_loyal)
    
    
    def prepare_specific_data_for_encoding(self):
        """Подготавливает данные таблицы data_newbies к кодированию признаков
        """
        # Добавляем новые признаки
        self.add_new_specific_features()
        
        # Сделаем признак contact бинарным (звонили по телефону или тип контакта неизвестен)
        self.data['contact'] = self.data['contact'].apply(lambda x: 1 if (x in ['cellular', 'telephone']) else 0)
        
        # Удаляем столбцы, которые больше не нужны
        self.data.drop(columns=self.columns_to_drop, inplace=True)
        
    
    def encoded_specific_columns(self):
        """Кодирует столбцы для таблицы data_newbies
        """
        # Подготавливаем данные для кодирования
        self.prepare_specific_data_for_encoding()
        
        # Кодируем данные
        self.encoded_columns()
        
    
    def get_specific_data_for_model(self):
        """Подготавливает и возвращает данные таблицы data_newbies для обработки моделью
        """
        # Кодируем признаки
        self.encoded_specific_columns()
        
        # Получим данные для обработки моделью
        X_train, X_test, y_train, y_test = self.get_common_data_for_model()
        
        return X_train, X_test, y_train, y_test
        
        