# Константы

from enum import Enum

# Имена полей. Поскольку будем часто использовать эту переменную, 
# назовем ее кратко F по первой букве слова Field
class F(Enum):
    LifeExpectancy = 'LifeExpectancy'
    SpatialDimValueCode = 'SpatialDimValueCode'
    Location = 'Location'
    Period = 'Period'
    Sanitation = 'Sanitation'
    GdpPerCapita = 'GdpPerCapita'
    NegativeCoef = 'NegativeCoef'
    ImmunizationMean = 'ImmunizationMean'
    SmoothingLifeExpectancy = 'SmoothingLifeExpectancy'

# Поля, которые будут использоваться для предсказания
prediction_fields = [
    F.ImmunizationMean.value,
    F.NegativeCoef.value,
    F.GdpPerCapita.value,
    F.Sanitation.value,
    F.SmoothingLifeExpectancy.value,
]

# Поля, значения которых можно менять через форму
form_fields = [
    F.Sanitation.value,
    F.ImmunizationMean.value,
    F.GdpPerCapita.value,
]

# Описание полей, которые можно менять через форму
form_field_descriptions = {
    F.Sanitation.value: 'Население, пользующееся как минимум базовыми услугами санитарии (%)',
    F.ImmunizationMean.value: 'Среднее значение охвата иммунизацией детей в возрасте 1 года от ' +
        'кори, полиомелита, дифтерии, столбняка и коколюша (%)',
    F.GdpPerCapita.value: 'ВВП на душу населения (в долл. США)',
}

# Максимально допустимые значения полей, которые можно менять через форму
form_field_max_values = {
    F.Sanitation.value: 100, # максимально 100, т.к. величина выражена в %
    F.ImmunizationMean.value: 100, # максимально 100, т.к. величина выражена в %
    F.GdpPerCapita.value: '', # максимальное значение не указываем, вышажено в долл. США
}

