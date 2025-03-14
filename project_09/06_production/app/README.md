С помощью данного функционала реализуется интерфейс для предсказания ожидаемой продолжительности в странах.

Исходные данные и модель собраны в проекте https://github.com/experiment0/sf_data_science/tree/main/project_09 .

Для запуска необходимо создать в данной папке виртуальную среду

`python -m venv app_venv`

Запустить ее

для windows: `./app_venv/Scripts/Activate.ps1` \
для unix: `source app_venv/bin/activate`

И установить библиотеки

`pip install -r requirements.txt`

И далее запустить файл сервера.

`python server.py`

Главная страница интерфейса должна отобразиться в браузере по ссылке http://localhost:5000/ .

Должен быть доступен интернет, так как страницы используют библиотеку bootstrap, которая подключается из внешнего источника.

Данный функционал помещен также в docker https://hub.docker.com/repository/docker/experiment0/life_expectancy/general

Для запуска необходимо выполнить команды

`docker pull experiment0/life_expectancy`

`docker run -it --rm --name=life_expectancy_container -p=5000:5000 experiment0/life_expectancy`

Открыть браузер по ссылке http://localhost:5000/⁠
