'''Guess the number game'''

import numpy as np


hidden_number = np.random.randint(1, 101)
attempts_count = 0


while True:
    attempts_count += 1
    predict_number = int(input('Guess a number from 1 to 100: '))
    
    if predict_number > hidden_number:
        print('Number must be less')
    elif predict_number < hidden_number:
        print('Number must be greater')
    else:
        print(f'Hidden number: {hidden_number}. Attempts count: {attempts_count}.')
        break
