"""Определение производительности функции, которая угадывает число."""

import numpy as np
import matplotlib.pyplot as plt

# Минимальное число для угадывания
min_number = 0
# Максимальное число для угадывания
max_number = 100
# Количество проверок на производительность
performance_checks_count = 1000


def guessing_number(hidden_number:int=1) -> int:
    """Угадывает число.

    Args:
        hidden_number(int, optional): Загаданное число. По умолчанию 1.

    Returns:
        int: Число попыток, за которое угадано число.
    """
    attempts_count = 0
    min_limit = min_number
    max_limit = max_number
    predict_number = None
    
    while not hidden_number == predict_number:
        attempts_count+=1

        if max_limit - min_limit > 1:
            predict_number = round((min_limit+max_limit)/2)
        elif min_limit == min_number:
            predict_number = min_limit
        elif max_limit == max_number:
            predict_number = max_limit
        
        if hidden_number > predict_number:
            min_limit = predict_number
        elif hidden_number < predict_number:
            max_limit = predict_number
            
    return(attempts_count)


def get_guess_function_performance(guess_function) -> int:
    """Определяет, за какое количество попыток в среднем
    переданная функция угадывает число.

    Args:
        guess_function(int, optional): Функция, которая угадывает число.

    Returns:
        int: Среднее количество попыток, 
        за которое переданная функция угадывает число.
    """
    # Фиксируем сид для производительности.
    np.random.seed(1)
    
    attempt_counts_list = []
    random_numbers_list = np.random.randint\
        (min_number, max_number + 1, size=(performance_checks_count))
    
    for random_number in random_numbers_list:
        attempt_counts_list.append(guess_function(random_number))
        
    mean_of_attempt_counts = int(np.mean(attempt_counts_list))
    print(f'Функция угадывает число в среднем \
        за {mean_of_attempt_counts} попыток.')
    
    return(mean_of_attempt_counts)


def get_count_of_divisions_by_2(number:int) -> int:
    """Определяет, сколько раз число делится на 2.

    Args:
        number(int): Число, для которого необходимо определить
        возможное количество делений.

    Returns:
        int: Возможное количество делений.
    """
    count_divisions = 0
    number_divisions_by_2 = number
    
    while not number_divisions_by_2 == 1:
        count_divisions += 1
        number_divisions_by_2 = round(number_divisions_by_2 / 2)
        
    return count_divisions


def researching_guess_function(guess_function):
    """Производит исследование функции, которая угадывает число.

    Args:
        guess_function(int, optional): Функция, которая угадывает число.
    """
    all_numbers_list = range(min_number, max_number + 1)
    attempt_counts_list = []
    
    for number in all_numbers_list:
        attempt_counts_list.append(guess_function(number))
    
    min_of_attempt_counts = min(attempt_counts_list)
    max_of_attempt_counts = max(attempt_counts_list)
    mean_of_attempt_counts = int(np.mean(attempt_counts_list))

    print(f'Минимальное количество попыток: {min_of_attempt_counts}')
    print(f'Максимальное количество попыток: {max_of_attempt_counts}')
    print(f'Среднее количество попыток: {mean_of_attempt_counts}')
    print()
    
    print(f'Число {max_number} можно разделить на два \
        {get_count_of_divisions_by_2(max_number)} раз.')
    print()
    
    numbers_by_attempt_counts_dict = \
        dict(map(
            lambda attempt_counts: (attempt_counts, []),
            range(min_of_attempt_counts, max_of_attempt_counts + 1)
        ))

    for number, attempt_counts in zip(all_numbers_list, attempt_counts_list):
        numbers_by_attempt_counts_dict[attempt_counts].append(number)

    for attempt_counts, numbers in  numbers_by_attempt_counts_dict.items():
        print(
            f'Количество попыток: {attempt_counts}',
            f'Угаданные за это количество попыток числа: {numbers}',
            f'Количество чисел: {len(numbers)}',
            '-----',
            sep='\n'
        )
    
    plt.title('График зависимости количества попыток угадывания от числа')
    plt.xlabel('Загаданные числа')
    plt.ylabel('Количество попыток угадывания')
    plt.plot(all_numbers_list, attempt_counts_list)
    plt.show()
    

if __name__ == '__main__': 
    get_guess_function_performance(guessing_number)
    researching_guess_function(guessing_number)
