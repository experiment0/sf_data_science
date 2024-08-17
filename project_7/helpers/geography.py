# Вспомогательные функции для расчета географических данных

import numpy as np


# Радиус Земли в километрах
EARTH_RADIUS = 6371


def get_haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Вычисляет расстояние между двумя точками по формуле гаверсинуса (в километрах)

    Args:
        lat1 (float): широта первой точки
        lng1 (float): долгота первой точки
        lat2 (float): широта второй точки
        lng2 (float): долгота второй точки

    Returns:
        float: расстояние между двумя точками (в километрах)
    """
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
         
    # считаем кратчайшее расстояние h по формуле гаверсинуса
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    
    return h


def get_angle_direction(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Вычисляет угол направления движения (в градусах)

    Args:
        lat1 (float): широта первой точки
        lng1 (float): долгота первой точки
        lat2 (float): широта второй точки
        lng2 (float): долгота второй точки

    Returns:
        float: угол направления движения (в градусах)
    """
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    
    # считаем угол направления движения alpha по формуле угла пеленга
    lng_delta_rad = lng2 - lng1
    
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    
    alpha = np.degrees(np.arctan2(y, x))
    
    return alpha