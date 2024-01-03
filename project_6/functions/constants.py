# Содержит константы общего назначения

from enum import Enum
from functions.helpers import get_year_xticks
from datetime import datetime

# Имя целевого признака
TARGET_FEATURE = 'deposit'

# Размер тестовой выборки
TEST_SIZE = 0.25

# Для воспроизводимости результатов случайных вычислений
RANDOM_STATE = 42

# Максимальное количество итераций, выделенное на сходимость
MAX_ITER = 1000

# Количество попыток для поиска лучших гиперпараметров
ATTEMPTS_COUNT = 100

# Количество фолдов для кросс-валидации
CV = 5

# Количество параллельных процессов (задаем -1, чтобы использовать все доступные ядра)
N_JOBS = -1

# Текущий год
CURRENT_YEAR = datetime.now().year

# Размер шрифта заголовков графиков
TITLE_FONT_SIZE = 14


# Группа поля
class FieldMeaning(Enum):
    # Целевая перемнная
    TARGET = 'target',
    # Поля с данными о клиентах банка
    CLIENT = 'client'
    # Поля с данными о текущей кампании банка
    CURRENT_PROMOTION = 'current_promotion'
    # Поля с данными о прошлой кампании банка
    PREVIOUS_PROMOTION = 'previous_promotion'


# Тип поля
class FieldType(Enum):
    CATEGORY = 'category',
    NUMERIC = 'numeric',
    BINARY = 'binary',
    DATETIME = 'datetime',


# Данные для полей таблицы
fields_params = {
    # Целевая переменная
    f'{TARGET_FEATURE}': {
        # Описание содержимого поля
        'description': 'открыл ли клиент депозит',
        # Тематика признака
        'field_meaning': FieldMeaning.TARGET.value,
        # Тип признака
        'field_type': FieldType.BINARY.value,
        # Параметры для графика распределения
        'distribution_graph_params': {
            'title': 'Распределение клиентов, открывших и не открывших депозит',
        },
    },
    # Данные о клиентах банка
    'balance': {
        'description': 'баланс',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.NUMERIC.value,
        'distribution_graph_params': {
            'title': 'Распределение баланса клиентов',
        },
        # Параметры для графика распределения по целевой переменной
        'distribution_by_target_graph_params': {
            'title': f'Распределение баланса клиентов в разрезе признака {TARGET_FEATURE}',
        },
    },
    'age': {
        'description': 'возраст',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.NUMERIC.value,
        'distribution_graph_params': {
            'title': 'Распределение возраста клиентов',
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение возраста клиентов в разрезе признака {TARGET_FEATURE}',
            'xticks': range(18, 95, 2),
            'binwidth': 0.5,
        },
    },
    'education': {
        'description': 'уровень образования',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по уровню образования',
            'figsize': (10, 5),
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по уровню образования в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
        },
    },
    'marital': {
        'description': 'семейное положение',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по семейному положению',
            'figsize': (10, 5),
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по семейному положению в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
        },
    },
    'job': {
        'description': 'сфера занятости',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по сферам занятости',
            'x_rotation': 30,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по сферам занятости в разрезе признака {TARGET_FEATURE}',
        },
    },
    'housing': {
        'description': 'имеется ли кредит на жильё',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.BINARY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по наличию кредита на жилье',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по наличию кредита на жилье в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
            'figsize': (5, 10),
        },
    },
    'loan': {
        'description': 'имеется ли кредит на личные нужды',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.BINARY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по наличию кредита на личные нужды',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по наличию кредита на личные нужды \
в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
            'figsize': (5, 10),
        },
    },
    'default': {
        'description': 'имеется ли просроченный кредит',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.BINARY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по наличию дефолта',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по наличию дефолта в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
            'figsize': (5, 10),
        },
    },
    # Данные, связанные с контактами в контексте текущей маркетинговой кампании
    'month': {
        'description': 'месяц, в котором был последний контакт',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Количество контактов для каждого месяца',
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по месяцу, в котором был последний контакт \
в разрезе признака {TARGET_FEATURE}',
        },
    },
    'day': {
        'description': 'день, в который был последний контакт',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Количество контактов для каждого дня',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по дню, в который был последний контакт \
в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
        },
    },
    'contact': {
        'description': 'тип контакта с клиентом',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по типу контакта',
            'figsize': (10, 5),
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по типу контакта в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
        },
    },
    'campaign': {
        'description': 'количество контактов с этим клиентом в течение текущей кампании',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.NUMERIC.value,
        'distribution_graph_params': {
            'title': 'Распределение количества контактов в течение текущей кампании',
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение количества контактов в течение текущей кампании \
в разрезе признака {TARGET_FEATURE}',
            'binwidth': 1,
        },
    },
    'duration': {
        'description': 'продолжительность контакта в секундах',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.NUMERIC.value,
        'distribution_graph_params': {
            'title': 'Распределение продолжительности контакта в секундах',
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение продолжительности контакта \
в разрезе признака {TARGET_FEATURE}',
        },
    },
    # Данные, связанные с контактами в контексте предыдущей маркетинговой кампании
    'previous': {
        'description': 'количество контактов до текущей кампании',
        'field_meaning': FieldMeaning.PREVIOUS_PROMOTION.value,
        'field_type': FieldType.NUMERIC.value,
        'distribution_graph_params': {
            'title': 'Распределение количества контактов до текущей кампании',
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение количества контактов до текущей кампании \
в разрезе признака {TARGET_FEATURE}',
            'binwidth': 1,
        },
    },
    'poutcome': {
        'description': 'результат прошлой маркетинговой кампании',
        'field_meaning': FieldMeaning.PREVIOUS_PROMOTION.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по результату прошлой маркетинговой кампании',
            'figsize': (10, 5),
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по результатам прошлой кампании \
в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
        },
    },
    'pdays': {
        'description': 'количество пропущенных дней с момента последней маркетинговой кампании \
до контакта в текущей кампании',
        'field_meaning': FieldMeaning.PREVIOUS_PROMOTION.value,
        'field_type': FieldType.NUMERIC.value,
        'distribution_graph_params': {
            'title': 'Распределение признака pdays',
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение признака pdays в разрезе признака {TARGET_FEATURE}',
            'binwidth': 1,
        },
    },
    # Новые признаки
    'has_income': {
        'description': 'имеется ли доход у клиента',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.BINARY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по наличию дохода',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по наличию дохода в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
            'figsize': (5, 10),
        },
    },
    'has_credit': {
        'description': 'имеется ли кредит у клиента',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.BINARY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по наличию кредита',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по наличию кредита в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
            'figsize': (5, 10),
        },
    },
    'is_debtor': {
        'description': 'отрицательный ли баланс у клиента',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.BINARY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по наличию отрицательного баланса',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по наличию отрицательного баланса в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
            'figsize': (5, 10),
        },
    },
    'age_scale': {
        'description': 'шкала возраста',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по шкале возраста',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по шкале возраста в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
        },
    },
    'is_working_age': {
        'description': 'является ли возраст клиента рабочим',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.BINARY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по рабочему возрасту',
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по рабочему возрасту в разрезе признака {TARGET_FEATURE}',
            'should_sort': False,
            'figsize': (5, 10),
        },
    },
    'duration_scale': {
        'description': 'шкала длительности последнего контакта',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по шкале длительности последнего контакта',
            'figsize': (10, 5),
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по шкале длительности последнего контакта в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
            'should_sort': False,
        },
    },
    'season': {
        'description': 'время года последнего контакта с клиентом',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по времени года последнего контакта',
            'figsize': (10, 5),
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по времени года последнего контакта в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
        },
    },    
    'contact_date': {
        'description': 'дата, когда был последний контакт',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.DATETIME.value,
        'distribution_graph_params': {
            'title': 'Количество контактов для каждой даты',
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов, открывших и не открывших {TARGET_FEATURE} по календарным датам',
            'xticks': get_year_xticks(),
            'x_rotation': 45,
            'binwidth': 2,
        },
    },
    'ringing_type': {
        'description': 'тип прозвона',
        'field_meaning': FieldMeaning.CURRENT_PROMOTION.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по типу прозвона последнего контакта',
            'figsize': (10, 5),
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по типу прозвона в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
        },
    },
    'job_loyal': {
        'description': 'уровень лояльности к предложению открыть депозит в зависимости от сферы занятости',
        'field_meaning': FieldMeaning.CLIENT.value,
        'field_type': FieldType.CATEGORY.value,
        'distribution_graph_params': {
            'title': 'Распределение клиентов по признаку job_loyal',
            'figsize': (10, 5),
            'should_sort': False,
        },
        'distribution_by_target_graph_params': {
            'title': f'Распределение клиентов по признаку job_loyal в разрезе признака {TARGET_FEATURE}',
            'figsize': (10, 5),
            'should_sort': False,
        },
    },
}
