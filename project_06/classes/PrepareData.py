import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

from functions.prepare_data import (
    get_balance,
    get_data_without_outliers,
    get_feature_scale,
    get_date,
    get_season,
    get_education_range,
    get_samples,
)

# Все новые признаки, которые создадим и исследуем
new_features = ['has_income', 'job_loyal',  'has_credit', 'is_debtor', 'age_scale', 
               'is_working_age', 'duration_scale', 'season', 'contact_date', 'ringing_type']


class PrepareData:
    def __init__(
            self, 
            data_source: pd.DataFrame, 
            new_features: list = new_features,
            columns_to_drop: list = [],
            best_features_count: int = 10,
        ) -> None:
        """Выполняет обработку данных,  общую для обеих таблиц (data_newbies и data_participated).

        Args:
            data_source (pd.DataFrame): исходная таблица с данными
            new_features (list, optional): список новых признаков. По умолчанию new_features.
            columns_to_drop (list, optional): список признаков для удаления перед кодированием. 
                                              По умолчанию []. Переопределяется для каждой таблицы.
            best_features_count (int, optional): Количество лучших признаков, которые оставляем. 
                                                 По умолчанию 10. Переопределяется для каждой таблицы.
        """
        # Делаем копию, чтобы не мутировать исходные данные
        self.data = data_source.copy()
        self.new_features = new_features
        self.columns_to_drop = columns_to_drop
        self.best_features_count = best_features_count
        
        
    def get_data(self) -> pd.DataFrame:
        """Возвращает таблицу с обработанными данными

        Returns:
            pd.DataFrame: таблица с обработанными данными
        """
        return self.data
    
    
    def update_data(self, data_source: pd.DataFrame):
        """Обновляет данные в объекте

        Args:
            data_source (pd.DataFrame): исходные данные
        """
        # Делаем копию, чтобы не мутировать исходные данные
        self.data = data_source.copy()
        
    
    def primary_prepare_data(self):
        """Производит первичную обработку данных перед разделением таблицы на 2 части
        """
        # Переведем признак с балансом в числовой
        self.data['balance'] = self.data['balance'].apply(get_balance)
        
        
    def set_specific_data():
        """Производит фильтрацию данных, специфическую для каждой таблицы (data_newbies и data_participated)
        """
        pass
    
    
    def common_prepare_data(self):
        """Производит предварительную обработку данных, общую для обеих таблиц
        (data_newbies и data_participated).
        Преобразует столбцы таблицы в нужный формат, заполняет пропуски и удаляет выбросы.
        Преобразования определены по результатам исследования.
        """
        # Заполним пропуски модой
        self.data['job'] = self.data['job'].apply(lambda value: 'management' if value == 'unknown' else value)
        self.data['education'] = self.data['education'].apply(lambda value: 'secondary' if value == 'unknown' else value)
        
        # Переведем бинарные признаки в числовые
        self.data['deposit'] = self.data['deposit'].apply(lambda value: 1 if value == 'yes' else 0)
        self.data['default'] = self.data['default'].apply(lambda value: 1 if value == 'yes' else 0)
        self.data['housing'] = self.data['housing'].apply(lambda value: 1 if value == 'yes' else 0)
        self.data['loan'] = self.data['loan'].apply(lambda value: 1 if value == 'yes' else 0)
        
        # Заполним пропуски признака balance медианой
        self.data['balance'] = self.data['balance'].fillna(self.data['balance'].median()) 
        
        # Удалим выбросы
        self.data = get_data_without_outliers(self.data, 'balance')
        
        
    def specific_prepare_data(self):
        """Производит предварительную обработку данных, специфическую для каждой таблицы 
        (data_newbies и data_participated)
        """
        pass
    

    def add_new_common_features(self):
        """Формирует в таблице новые признаки из списка self.new_features
        """
        if ('has_income' in self.new_features):
            # Имеется ли доход у клиента
            self.data['has_income'] = self.data['job'].apply(
                lambda x: 0 if (x in ['retired', 'student', 'unemployed']) else 1
            )
        
        if ('has_credit' in self.new_features):
            # Имеется ли кредит у клиента
            self.data['has_credit'] = self.data.apply(
                lambda x: 1 if (x['housing'] or x['loan'] or x['default']) else 0, axis=1
            )
        
        if ('is_debtor' in self.new_features):
            # Отрицательный ли баланс у клиента
            self.data['is_debtor'] = self.data['balance'].apply(lambda x: 1 if (x < 0) else 0)
            
        if ('age_scale' in self.new_features):
            # Шкала возраста
            age_thresholds = range(0, 110, 10)
            self.data['age_scale'] = self.data['age'].apply(lambda x: get_feature_scale(x, age_thresholds))
        
        if ('is_working_age' in self.new_features):
            # Является ли возраст клиента рабочим
            self.data['is_working_age'] = self.data['age'].apply(lambda x: 1 if (x >= 25 and x <= 60) else 0)
        
        if ('duration_scale' in self.new_features):
            # Шкала длительности последнего контакта
            duration_thresholds = range(0, 2000, 600)
            self.data['duration_scale'] = self.data['duration'].apply(
                lambda x: get_feature_scale(x, duration_thresholds)
            )
        
        if ('season' in self.new_features):
            # Время года последнего контакта с клиентом
            self.data['season'] = self.data['month'].apply(get_season)
        
        if ('contact_date' in self.new_features):
            # Дата, когда был последний контакт
            self.data['contact_date'] = self.data.apply(lambda x: get_date(x['month'], x['day']), axis=1)
            

    def add_new_specific_features():
        """Добавляет новые признаки из списка self.new_features, 
        специфически сформированные для каждой таблицы (data_newbies и data_participated)
        """
        pass
    
    
    def prepare_specific_data_for_encoding():
        """Подготавливает данные таблиц к кодированию признаков для каждой из таблиц
        (data_newbies и data_participated)
        """
        pass
    
    
    def encoded_columns(self):
        """Производит кодирование столбцов
        """
        # Список столбцов таблицы
        columns = self.data.columns.to_list()
        
        if ('education' in columns):
            # Переведем признак education в порядковый
            self.data['education'] = self.data['education'].apply(get_education_range)
        
        if ('marital' in columns):
            # Сделаем признак marital бинарным (в отношениях или нет)
            self.data['marital'] = self.data['marital'].apply(lambda x: 1 if (x == 'married') else 0)
        
        # Производим однократное кодирование
        self.data = pd.get_dummies(self.data)
        
        # Задаем столбцы для бинарного кодирования
        columns_for_be = []
        if ('contact_date' in columns):
            columns_for_be.append('contact_date')
        if ('day' in columns):
            columns_for_be.append('day')
        
        # Если в таблице есть столбцы для бинарного кодирования
        if (len(columns_for_be)):
            # Переведем эти столбцы в строковый тип
            for column in columns_for_be:
                self.data[column] = self.data[column].astype(str)
            
            # Производим бинарное кодирование
            binary_encoder = ce.BinaryEncoder(cols=columns_for_be)
            binary_encoded_columns = binary_encoder.fit_transform(self.data[columns_for_be])
            
            # Объединяем закодированные столбцы с остальными
            self.data = pd.concat([self.data, binary_encoded_columns], axis=1)
            
            # Удаляем исходные столбцы
            self.data.drop(columns=columns_for_be, inplace=True)
            
    
    def encoded_specific_columns():
        """Кодирует столбцы для каждой таблицы (data_newbies и data_participated)
        """
        pass
    
    
    def get_common_data_for_model(self):
        """Выполняет преобразования данных, необходимые для передачи в модель
        """
        # Получим данные тренировочной и тестовой выборок
        X_train, X_test, y_train, y_test = get_samples(self.data)
        
        # Определим наиболее ценные признаки
        selector = SelectKBest(score_func=f_classif, k=self.best_features_count)
        selector.fit(X_train, y_train)
        best_features = selector.get_feature_names_out()
        
        # Выделим лучше признаки из выборок
        X_train = X_train[best_features]
        X_test = X_test[best_features]
        
        # Для нормализации признаков воспользуемся MinMaxScaler
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(X_train)
        X_train_scaled = min_max_scaler.transform(X_train)
        X_test_scaled = min_max_scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    
    def get_specific_data_for_model():
        """Подготавливает и возвращает данные для обработки моделью
        для каждой таблицы (data_newbies и data_participated)
        """
        pass