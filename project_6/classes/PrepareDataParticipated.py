from datetime import datetime
from pandas import DataFrame

from classes.PrepareData import PrepareData, new_features
from functions.constants import (
    CURRENT_YEAR, TARGET_FEATURE, RANDOM_STATE, TEST_SIZE
)

# Столбцы для удаления из таблицы data_participated
columns_to_drop = ['age', 'housing', 'loan', 'default', 'duration_scale', 'season', 'month', 
                   'has_income', 'job', 'contact_date']

# Количество признаков, которые оставим в таблице data_participated
best_features_count = 20


class PrepareDataParticipated(PrepareData):
    def __init__(
        self, 
        data_source: DataFrame, 
        new_features: list = new_features, 
        columns_to_drop: list = columns_to_drop,
        best_features_count: int = best_features_count,
    ) -> None:
        """Выполняет специфические преобразования для таблицы data_participated

        Args:
            data_source (DataFrame):  исходная таблица с данными
            new_features (list, optional): список новых признаков. По умолчанию new_features.
            columns_to_drop (list, optional): список признаков для удаления перед кодированием. 
                                              По умолчанию columns_to_drop.
            best_features_count (int, optional): Количество лучших признаков, которые оставляем. 
                                                 По умолчанию best_features_count.
        """
        super().__init__(data_source, new_features, columns_to_drop, best_features_count)
    
    
    def set_specific_data(self):
        """Формирует таблицу с клиентами, участвовавшими ранее в кампаниях банка.
           Преобразования определены по результатам исследования.
        """
        self.primary_prepare_data()
        
        # В признаке `previous` содержится количество контактов до текущей кампании.
        # Если оно равно 0, значит текущая кампания для клиента первая.
        mask_not_participated = self.data['previous'] == 0
        
        # Удаляем строки по маске
        self.data.drop(index=self.data[mask_not_participated].index, axis=0, inplace=True)
    
    
    def specific_prepare_data(self):
        """Выполняет предварительное преобразование данных для таблицы data_participated.
        Преобразования определены по результатам исследования.
        """
        # Фильтруем данные, специфические для данной таблицы
        self.set_specific_data()
        
        # Делаем общие преобразования
        self.common_prepare_data()
        
        # Заполним пропуски
        self.data['contact'] = \
            self.data['contact'].apply(
                lambda value: 'cellular' if value == 'unknown' else value
            )
            
        self.data['poutcome'] = \
            self.data['poutcome'].apply(
                lambda value: 'other' if value == 'unknown' else value
            )
    
    
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

        if (
            # 16 апреля - 20 апреля
            (date >= get_date(16, 4) and date <= get_date(20, 4)) or
            # 4 мая - 18 мая
            (date >= get_date(4, 5) and date <= get_date(18, 5)) or
            # 18 ноября - 21 ноября
            (date >= get_date(18, 11) and date <= get_date(21, 11))
        ):
            # Прозвон по кредиту на жилье
            return 'housing'
        # 30 апреля
        if (date == get_date(30, 4)):
            # Прозвон по депозиту
            return 'deposit'
        if (
            # 28 января - 9 февраля
            (date >= get_date(28, 1) and date <= get_date(9, 2)) or
            # 13 апреля - 20 апреля
            (date >= get_date(13, 4) and date <= get_date(20, 4)) or
            # 4 мая - 18 мая
            (date >= get_date(4, 5) and date <= get_date(18, 5)) or
            # 16 ноября - 21 ноября
            (date >= get_date(16, 11) and date <= get_date(21, 11))
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
        if (job in ['blue-collar', 'entrepreneur']):
            return 1
        if (job in ['housemaid', 'self-employed', 'services', 'admin.', 'technician']):
            return 2
        if (job == 'management'):
            return 3
        if (job in ['retired', 'student', 'unemployed']):
            return 4
        
        
    def add_new_specific_features(self):
        """Добавляет новые признаки из списка self.new_features, 
        специфически сформированные для таблицы data_participated
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
        """Подготавливает данные таблицы data_participated к кодированию признаков
        """
        # Добавляем новые признаки
        self.add_new_specific_features()
        
        self.data['contact'] = self.data['contact'].apply(lambda x: 1 if (x == 'cellular') else 0)
        
        # Удаляем столбцы, которые больше не нужны
        self.data.drop(columns=self.columns_to_drop, inplace=True)
    
    
    def encoded_specific_columns(self):
        """Кодирует столбцы для таблицы data_participated
        """
        # Подготавливаем данные для кодирования
        self.prepare_specific_data_for_encoding()
        
        # Кодируем данные
        self.encoded_columns()
    
    
    def get_specific_data_for_model(self):
        """Подготавливает и возвращает данные таблицы data_participated для обработки моделью
        """
        # Кодируем признаки
        self.encoded_specific_columns()
        
        # Получим данные для обработки моделью
        X_train, X_test, y_train, y_test = self.get_common_data_for_model()
        
        return X_train, X_test, y_train, y_test