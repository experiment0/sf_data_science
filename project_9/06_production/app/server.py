from flask import Flask, render_template, request
import pickle

from utils.constants import (
    F,
    prediction_fields,
    form_field_descriptions,
    form_field_max_values,
)
from utils.prepare_data import (
    get_locations,
    get_location_by_code,
    get_form_predictors,
    get_updated_location_future_data,
    get_location_past_data,
)
from utils.graphics import (
    create_prediction_graphic,
)

app = Flask(__name__)

# Десериализуем pipeline для предсказания из файла
with open('./model/pipeline.pkl', 'rb') as pkl_file:
    prediction_pipeline = pickle.load(pkl_file)


# Главная страница с формой выбора страны, для которой нужно сделать предсказание
@app.route('/', methods=['GET'])
def select_location():
    # Получаем список названий стран и их кодов
    locations = get_locations()
    return render_template('select_location.html', locations=locations)


# Страница с формой переопределения предикторов 
# (тех, которые изначально определены с помощью сглаживания значений за предыдущие года)
@app.route('/set_predictors', methods=['GET', 'POST'])
def set_predictors():
    # Код страны, выбранной на предыдущем шаге
    location_code = request.form['location_code']
    # Название выбранной страны
    location = get_location_by_code(location_code)
    # Данные для предзаполнения формы ранее посчитанными значениями для будущих лет
    form_years, form_predictors = get_form_predictors(location_code)
    
    return render_template(
        'set_predictors.html',
        location_code=location_code,
        location=location,
        form_years=form_years,
        form_predictors=form_predictors,
        form_field_descriptions=form_field_descriptions,
        form_field_max_values=form_field_max_values,
    )


# Страница с выводом результата предсказания
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    # Данные страны для будущих лет, переопределенные пользователем
    location_future_data = get_updated_location_future_data(request.form)
    # Предсказание ожидаемой продолжительности жизни для будущих лет
    predict = prediction_pipeline.predict(location_future_data[prediction_fields])
    # Заполним столбец с продолжительностью жизни предсказанием (нужно для построения графика)
    location_future_data[F.LifeExpectancy.value] = predict
    
    # Года, за которые сделали предсказание
    years = location_future_data[F.Period.value].values
    # Собираем словарь в формате: год: предсказанное_значение
    predict_by_years = dict(zip(years, predict))
    
    # Данные рассматриваемой страны за прошлые года (нужно для построения графика)
    location_past_data = get_location_past_data(request.form['location_code'])
    
    # Строим график со значениями продолжительности жизни за прошлые года и значениями прогноза
    # И сохраняем его в виде картинки для вывода на данной странице.
    create_prediction_graphic(
        location_past_data,
        location_future_data,
    )
    
    return render_template(
        'prediction.html',
        predict_by_years=predict_by_years,
    )


if __name__ == '__main__':
    app.run('localhost', 5000)