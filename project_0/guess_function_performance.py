"""Определение производительности функции, которая рандомно угадывает число."""

import numpy as np


# Минимальное число для угадывания
min_number = 1
# Максимальное число для угадывания
max_number = 10


def get_count_attempts_guess_number(hidden_number:int=1) -> int:
    """ Рандомно угадывает число.

    Args:
        hidden_number (int, optional): Загаданное число. По умолчанию 1.

    Returns:
        int: Число попыток, за которое угадано число.
    """
    attempts_count = 0
    
    while True:
        attempts_count+=1
        predict_number = np.random.randint(min_number, max_number + 1)
        if hidden_number == predict_number:
            break

    return(attempts_count)


def get_guess_function_performance(guess_function) -> int:
    """За какое количество попыток в среднем функция угадывает число.

    Args:
        guess_function: Функция, которая угадывает число.

    Returns:
        int: Среднее количество попыток, за которое переданная функция угадывает число.
    """
    
    # фиксируем сид для производительности
    np.random.seed(1)
    
    attempt_counts_list = []
    random_numbers_list = np.random.randint(min_number, max_number + 1, size=(max_number))
    
    for random_number in random_numbers_list:
        attempt_counts_list.append(guess_function(random_number))
        
    mean_of_attempt_counts = int(np.mean(attempt_counts_list))
    print(f'Функция угадывает число в среднем за {mean_of_attempt_counts} попыток.')
    
    return(mean_of_attempt_counts)

if __name__ == '__main__': 
    get_guess_function_performance(get_count_attempts_guess_number)
