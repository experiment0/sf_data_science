# Константы

from enum import Enum


# Значение для воспроизводимости генерации случайных процессов
RANDOM_STATE = 42

# Количество лет для тестовой выборки
TEST_YEARS_COUNT = 3

# Название алгоритма кластеризации
class ClusteringAlgorithm(Enum):
    K_MEANS = 'k_means'
    EM = 'em'
    DBSCAN = 'dbscan'
    AC = 'agglomerative_clustering'

# Размер шрифта заголовков графиков
TITLE_FONT_SIZE = 14


# Смысловые категории полей таблицы
class FieldSemanticCategory(Enum):
    TARGET = 'target'
    TIME = 'time'
    COUNTRY_NAMING = 'country_naming'
    COUNTRY_INDICATORS = 'country_indicators'
    MEDICINE = 'medicine'
    IMMUNIZATION = 'immunization'
    CHILD_MORTALITY = 'child_mortality'
    ADULT_MORTALITY = 'adult_mortality'
    HEALTH = 'health'
    DISEASES = 'diseases'


# Описание смысловых категорий полей
field_semantic_category_descriptions = {
    FieldSemanticCategory.TARGET.value: 
        'Средняя продолжительность жизни в стране (целевая переменная)',
    FieldSemanticCategory.TIME.value: 
        'Время измерений',
    FieldSemanticCategory.COUNTRY_NAMING.value: 
        'Названия и коды страны и ее региона',
    FieldSemanticCategory.COUNTRY_INDICATORS.value:
        'Основные показатели страны (ВВП, численность населения и пр.)',
    FieldSemanticCategory.MEDICINE.value:
        'Факторы, связанные с медициной и санитарией в стране',
    FieldSemanticCategory.IMMUNIZATION.value: 
        'Иммунизация населения различными способами',
    FieldSemanticCategory.CHILD_MORTALITY.value:
        'Коэффициенты детской смертности от разных причин',
    FieldSemanticCategory.ADULT_MORTALITY.value: 
        'Коэффициенты взрослой смертности от разных причин',
    FieldSemanticCategory.HEALTH.value: 
        'Показатели здорового (или нет) образа жизни населения',
}


# Категория происхождения поля (из какой таблицы взято или сгенерировано из других полей)
class FieldOriginCategory(Enum):
    # Поле из базовой таблицы
    BASE = 'base'
    # Поле из таблицы для исследования
    FOR_RESEARCH = 'for_research'
    # Сгенерированные признаки
    GENERATED = 'generated'
    

# Имена полей. Поскольку будем часто использовать эту переменную, 
# назовем ее кратко F по первой букве слова Field
class F(Enum):
    LifeExpectancy = 'LifeExpectancy'
    ParentLocationCode = 'ParentLocationCode'
    ParentLocation = 'ParentLocation'
    SpatialDimValueCode = 'SpatialDimValueCode'
    Location = 'Location'
    Period = 'Period'
    AdultMortality = 'AdultMortality'
    ChildUnder5Mortality2 = 'ChildUnder5Mortality2'
    ChildUnder5Mortality3 = 'ChildUnder5Mortality3'
    ChildUnder5Mortality5 = 'ChildUnder5Mortality5'
    ChildUnder5Mortality6 = 'ChildUnder5Mortality6'
    ChildUnder5Mortality7 = 'ChildUnder5Mortality7'
    ChildUnder5Mortality8 = 'ChildUnder5Mortality8'
    ChildUnder5Mortality9 = 'ChildUnder5Mortality9'
    ChildUnder5Mortality10 = 'ChildUnder5Mortality10'
    ChildUnder5Mortality11 = 'ChildUnder5Mortality11'
    ChildUnder5Mortality12 = 'ChildUnder5Mortality12'
    ChildUnder5Mortality13 = 'ChildUnder5Mortality13'
    ChildUnder5Mortality15 = 'ChildUnder5Mortality15'
    ChildUnder5Mortality16 = 'ChildUnder5Mortality16'
    ChildUnder5Mortality17 = 'ChildUnder5Mortality17'
    Homicides = 'Homicides'
    MaternalMortality = 'MaternalMortality'
    AdultNcdMortality = 'AdultNcdMortality'
    AdultNcdMortality061 = 'AdultNcdMortality061'
    AdultNcdMortality080 = 'AdultNcdMortality080'
    AdultNcdMortality110 = 'AdultNcdMortality110'
    AdultNcdMortality117 = 'AdultNcdMortality117'
    AdultNcdMortalitySum = 'AdultNcdMortalitySum'
    PoisoningMortality = 'PoisoningMortality'
    SuicideMortality = 'SuicideMortality'
    AlcoholСonsumption = 'AlcoholСonsumption'
    HepatitisBImmunization = 'HepatitisBImmunization'
    MeaslesImmunization = 'MeaslesImmunization'
    PolioImmunization = 'PolioImmunization'
    DiphtheriaImmunization = 'DiphtheriaImmunization'
    BmiAdultUnderweight = 'BmiAdultUnderweight'
    BmiAdultOverweight25 = 'BmiAdultOverweight25'
    BmiAdultOverweight30 = 'BmiAdultOverweight30'
    BmiChildThinness = 'BmiChildThinness'
    BmiTeenagerThinness = 'BmiTeenagerThinness'
    BmiChildOverweight1 = 'BmiChildOverweight1'
    BmiTeenagerOverweight1 = 'BmiTeenagerOverweight1'
    BmiChildOverweight2 = 'BmiChildOverweight2'
    BmiTeenagerOverweight2 = 'BmiTeenagerOverweight2'
    Sanitation = 'Sanitation'
    DrinkingWater = 'DrinkingWater'
    HealthCareCosts = 'HealthCareCosts'
    HealthCareCostsGdp = 'HealthCareCostsGdp'
    HealthCareCostsPerCapita = 'HealthCareCostsPerCapita'
    GdpPerCapita = 'GdpPerCapita'
    Population = 'Population'
    Schooling = 'Schooling'    
    ClusterKMeans = 'ClusterKMeans'
    PositiveCoef = 'PositiveCoef'
    NegativeCoef = 'NegativeCoef'
    ImmunizationMean = 'ImmunizationMean'
    SmoothingLifeExpectancy = 'SmoothingLifeExpectancy'


# Имя целевого признака 
# (константа нужна для удобства переиспользования кода с участием целевой переменной)
TARGET_FEATURE = F.LifeExpectancy.value


# Данные полей
fields = {
    # Ключ - это имя поля в таблице
    F.LifeExpectancy.value: {
        # Описание содержимого поля
        'description': 'Ожидаемая продоложительность жизни',
        # Смысловая категория поля
        'semantic_category': FieldSemanticCategory.TARGET.value,
        # Категория происхождения поля (из какой таблицы или как сгенерировано и пр.)
        'origin_category': FieldOriginCategory.BASE.value,
        # Флаг, является ли поле предиктором
        'is_predictor': False,
    },
    F.ParentLocationCode.value: {
        'description': 'Код региона',
        'semantic_category': FieldSemanticCategory.COUNTRY_NAMING.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.ParentLocation.value: {
        'description': 'Название региона',
        'semantic_category': FieldSemanticCategory.COUNTRY_NAMING.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.SpatialDimValueCode.value: {
        'description': 'Код страны',
        'semantic_category': FieldSemanticCategory.COUNTRY_NAMING.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.Location.value: {
        'description': 'Название страны',
        'semantic_category': FieldSemanticCategory.COUNTRY_NAMING.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.Period.value: {
        'description': 'Год',     
        # Текст для временного переименования признака для удобства чтения графиков   
        'temp_rename': 'Год',
        'semantic_category': FieldSemanticCategory.TIME.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.AdultMortality.value: {
        # Описание поля с единицей измерения, полученной из исходной таблицы
        'description_source': 'Коэффициент смертности среди взрослого населения '+
            '(вероятность смерти в возрасте от 15 до 60 лет на 1000 человек населения).',
        # Описание поля с единицей измерения после преобразования
        'description': 'Смертность среди взрослого населения от 15 до 60 лет (в % от численности населения)',
        'temp_rename': 'Смертность в 15-60 лет',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        # Факторы смертности не будем делать предикторами, т.к. из них формируется
        # значение ожидаемой продолжительности жизни. Чтобы не было утечки данных.
        # Но исследуем эти данные.
        'is_predictor': False,
    },
    F.ChildUnder5Mortality2.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). ВИЧ/СПИД.',
        # Буква "Д" в начале означает "Дети", цира 2 - номер признака из исходного названия.
        'temp_rename': 'Д-2. ВИЧ/СПИД',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality3.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Диарейные заболевания.',
        'temp_rename': 'Д-3. Диарея',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality5.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). Столбняк.',
        'temp_rename': 'Д-5. Столбняк',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality6.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). Корь.',
        'temp_rename': 'Д-6. Корь',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality7.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Менингит/энцефалит.',
        'temp_rename': 'Д-7. Менингит/энцефалит',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality8.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). Малярия.',
        'temp_rename': 'Д-8. Малярия',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality9.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Острые инфекции нижних дыхательных путей.',
        'temp_rename': 'Д-9. Острые инф. НДП',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality10.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Недоношенность.',
        'temp_rename': 'Д-10. Недоношенность',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality11.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Родовая асфиксия и родовая травма.',
        'temp_rename': 'Д-11. Родовая травма',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality12.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Сепсис и другие инфекционные состояния новорожденных.',
        'temp_rename': 'Д-12. Сепсис и др. инф.',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality13.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Другие инфекционные, перинатальные и алиментарные состояния.',
        'temp_rename': 'Д-13. Другое',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality15.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Врожденные аномалии.',
        'temp_rename': 'Д-15. Врожденные аномалии',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality16.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). '+
            'Другие неинфекционные заболевания.',
        'temp_rename': 'Д-16. Другие неинф. заб.',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.ChildUnder5Mortality17.value: {
        'description': 'Распределение причин смерти среди детей в возрасте до 5 лет (%). Травмы.',
        'temp_rename': 'Д-17. Травмы',
        'semantic_category': FieldSemanticCategory.CHILD_MORTALITY.value,
        'origin_category': FieldOriginCategory.FOR_RESEARCH.value,
        'is_predictor': False,
    },
    F.Homicides.value: {
        'description_source': 'Оценка уровня убийств (на 100 000 населения)',
        'description': 'Оценка уровня убийств (%)',
        'temp_rename': 'Оценка уровня убийств',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.MaternalMortality.value: {
        'description_source': 'Коэффициент материнской смертности (на 100 000 живорождений)',
        'description': 'Коэффициент материнской смертности (%)',
        'temp_rename': 'Материнская смертность',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.AdultNcdMortality.value: {
        'description': 'Вероятность (в %) смерти в возрасте от 30 до 70 лет '+
            'от сердечно-сосудистых заболеваний, рака, диабета или хронических респираторных заболеваний',
        # Сокращение НЗ - неинфекционные заболевания
        'temp_rename': 'Смертность в 30-70 лет от НЗ',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.AdultNcdMortality061.value: {
        'description_source': 'Причина смертности взрослых от неинфекционных заболеваний. '+
            'Злокачественные новообразования',
        'description': 'Смертность взрослых от неинфекционных заболеваний (%). '+
            'Злокачественные новообразования',
        # Буква "В" в начале означает "Взрослые", цира 061 - номер признака из исходного названия.
        'temp_rename': 'В-061. Злокач. новообр.',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.AdultNcdMortality080.value: {
        'description_source': 'Причина смертности взрослых от неинфекционных заболеваний. '+
            'Сахарный диабет',
        'description': 'Смертность взрослых от неинфекционных заболеваний (%). '+
            'Сахарный диабет',
        'temp_rename': 'В-080. Сахарный диабет',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.AdultNcdMortality110.value: {
        'description_source': 'Причина смертности взрослых от неинфекционных заболеваний. '+
            'Сердечно-сосудистые заболевания',
        'description': 'Смертность взрослых от неинфекционных заболеваний (%). '+
            'Сердечно-сосудистые заболевания',
        'temp_rename': 'В-110. Сердечно-сосуд. заб.',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.AdultNcdMortality117.value: {
        'description_source': 'Причина смертности взрослых от неинфекционных заболеваний. '+
            'Респираторные заболевания',
        'description': 'Смертность взрослых от неинфекционных заболеваний (%). '+
            'Респираторные заболевания',
        'temp_rename': 'В-117. Респират. заб.',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.AdultNcdMortalitySum.value: {
        'description_source': 'Сумма причин смертности взрослых от неинфекционных заболеваний.',
        'description': 'Сумма причин (в %) смертности взрослых от неинфекционных заболеваний.',
        'temp_rename': 'В. Сумма смертности от НЗ',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.PoisoningMortality.value: {
        'description_source': 'Уровень смертности от непреднамеренного отравления (на 100 000 населения)',
        'description': 'Уровень смертности от непреднамеренного отравления (%)',
        'temp_rename': 'Отравления',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.SuicideMortality.value: {
        'description_source': 'Стандартизированные по возрасту показатели самоубийств (на 100 000 населения)',
        'description': 'Стандартизированные по возрасту показатели самоубийств (%)',
        'temp_rename': 'Самоубийства',
        'semantic_category': FieldSemanticCategory.ADULT_MORTALITY.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': False,
    },
    F.AlcoholСonsumption.value: {
        'description': 'Потребление алкоголя на душу населения (15+) (в литрах чистого алкоголя)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.HepatitisBImmunization.value: {
        'description': 'Охват иммунизацией детей в возрасте 1 года против гепатита В (ГепВ3) (%)',
        'temp_rename': 'Имм. от гепатита В',
        'semantic_category': FieldSemanticCategory.IMMUNIZATION.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.MeaslesImmunization.value: {
        'description': 'Охват первой дозой вакцины, содержащей коревой компонент (MCV1), '+
            'среди детей в возрасте 1 года (%)',
        'temp_rename': 'Имм. от кори',
        'semantic_category': FieldSemanticCategory.IMMUNIZATION.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.PolioImmunization.value: {
        'description': 'Охват иммунизацией от полиомиелита (Pol3) среди детей в возрасте 1 года (%)',
        'temp_rename': 'Имм. от полиомиелита',
        'semantic_category': FieldSemanticCategory.IMMUNIZATION.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.DiphtheriaImmunization.value: {
        'description': 'Охват иммунизацией детей в возрасте 1 года '+
            'дифтерийно-столбнячным анатоксином и коклюшем (АКДС3) (%)',
        'temp_rename': 'Имм. дифтерия, столбняк, коклюш',
        'semantic_category': FieldSemanticCategory.IMMUNIZATION.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiAdultUnderweight.value: {
        'description': 'Распространенность недостаточного веса среди взрослых, '+
            'ИМТ < 18,5 (стандартизированная по возрасту оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiAdultOverweight25.value: {
        'description': 'Распространенность избыточного веса среди взрослых, '+
            'ИМТ >= 25 (стандартизированная по возрасту оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiAdultOverweight30.value: {
        'description': 'Распространенность ожирения среди взрослых, '+
            'ИМТ >= 30 (стандартизированная по возрасту оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiChildThinness.value: {
        'description': 'Распространенность худобы среди детей 5-9 лет, '+
            'ИМТ < -2 стандартных отклонений ниже медианы (грубая оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiTeenagerThinness.value: {
        'description': 'Распространенность худобы среди подростков 10-19 лет, '+
            'ИМТ < -2 стандартных отклонений ниже медианы (грубая оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiChildOverweight1.value: {
        'description': 'Распространенность избыточного веса среди детей 5-9 лет, '+
            'ИМТ > +1 стандартного отклонения выше медианы (грубая оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiTeenagerOverweight1.value: {
        'description': 'Распространенность избыточного веса среди подростков 10-19 лет, '+
            'ИМТ > +1 стандартного отклонения выше медианы (грубая оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiChildOverweight2.value: {
        'description': 'Распространенность ожирения среди детей 5-9 лет, '+
            'ИМТ > +2 стандартных отклонений выше медианы (грубая оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.BmiTeenagerOverweight2.value: {
        'description': 'Распространенность ожирения среди подростков 10-19 лет, '+
            'ИМТ > +2 стандартных отклонений выше медианы (грубая оценка) (%)',
        'semantic_category': FieldSemanticCategory.HEALTH.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.Sanitation.value: {
        'description': 'Население, пользующееся как минимум базовыми услугами санитарии (%)',
        'semantic_category': FieldSemanticCategory.MEDICINE.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.DrinkingWater.value: {
        'description': 'Население, пользующееся по крайней мере базовыми услугами питьевого водоснабжения (%)',
        'semantic_category': FieldSemanticCategory.MEDICINE.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.HealthCareCosts.value: {
        'description': 'Общие внутренние государственные расходы на здравоохранение '+
            'в процентах от общих государственных расходов (%)',
        'semantic_category': FieldSemanticCategory.MEDICINE.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.HealthCareCostsGdp.value: {
        'description': 'Общие внутренние государственные расходы на здравоохранение '+
            'в процентах от валового внутреннего продукта (ВВП) (%)',
        'semantic_category': FieldSemanticCategory.MEDICINE.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.HealthCareCostsPerCapita.value: {
        'description': 'Общие внутренние государственные расходы на здравоохранение '+
            'на душу населения в долларах США',
        'semantic_category': FieldSemanticCategory.MEDICINE.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.GdpPerCapita.value: {
        'description': 'ВВП на душу населения (в долл. США)',
        'semantic_category': FieldSemanticCategory.COUNTRY_INDICATORS.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.Population.value: {
        'description': 'Общая численность населения',
        'semantic_category': FieldSemanticCategory.COUNTRY_INDICATORS.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.Schooling.value: {
        'description': 'Средняя продолжительность формального образования для лиц в возрасте 15–64 лет',
        'semantic_category': FieldSemanticCategory.COUNTRY_INDICATORS.value,
        'origin_category': FieldOriginCategory.BASE.value,
        'is_predictor': True,
    },
    F.ClusterKMeans.value: {
        'description': 'Кластер страны, определенный с помощью k-means',
        'origin_category': FieldOriginCategory.GENERATED.value,
        'is_predictor': True,
    },
    F.PositiveCoef.value: {
        'description': 'Коэффициент благополучия страны',
        'origin_category': FieldOriginCategory.GENERATED.value,
        'is_predictor': True,
    },
    F.NegativeCoef.value: {
        'description': 'Коэффициент неблагополучия страны',
        'origin_category': FieldOriginCategory.GENERATED.value,
        'is_predictor': True,
    },
    F.ImmunizationMean.value: {
        'description': 'Среднее значение охвата иммунизацией детей в возрасте 1 года '+
            'от кори, полиомелита, дифтерии, столбняка и коколюша (%)',
        'origin_category': FieldOriginCategory.GENERATED.value,
        'is_predictor': True,
    },
    F.SmoothingLifeExpectancy.value: {
        'description': 'Значения, полученные в результате использования экспоненциального сглаживания. '+
            'Тренировочные данные получены в ходе обучения модели. Тестовые получены в качестве прогноза.',
        'origin_category': FieldOriginCategory.GENERATED.value,
        'is_predictor': True,
    }
}