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
    PositiveCoef = 'PositiveCoef'
    ImmunizationMean = 'ImmunizationMean'
    SmoothingLifeExpectancy = 'SmoothingLifeExpectancy'
    Reg_AFR = 'Reg_AFR'
    Reg_EUR = 'Reg_EUR'

# Поля, которые будут использоваться для предсказания
prediction_fields = [
    F.Sanitation.value,
    F.GdpPerCapita.value,
    F.PositiveCoef.value,
    F.NegativeCoef.value,
    F.ImmunizationMean.value,
    F.SmoothingLifeExpectancy.value,
    F.Reg_AFR.value,
    F.Reg_EUR.value,
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
    F.GdpPerCapita.value: '', # максимальное значение не указываем, выражено в долл. США
}
