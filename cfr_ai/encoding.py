import numpy as np

encoding_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY'
digit_strings = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def clear_lows(probabilities: np.ndarray) -> np.ndarray:
    """Return a copy with sub-1% entries zeroed and the rest renormalised."""
    cleaned = np.where(probabilities < 0.01, 0.0, probabilities)
    s = cleaned.sum()
    return cleaned / s if s > 0 else cleaned


def encode_probabilities(probabilities: np.ndarray) -> str:
    concatenated = ''
    zero_counter = 0
    for p in probabilities:
        inflated = int(round(p * 2500)) # Round instead of truncating to not bias the probabilites
        if inflated == 0:
            zero_counter += 1
        else:
            if zero_counter > 0:
                zeros = str(int(zero_counter / 10)) + str(zero_counter % 10)
                concatenated += zeros
                zero_counter = 0
            part_1 = int(inflated / 50)
            part_2 = inflated % 50
            concatenated += encoding_characters[part_1] + encoding_characters[part_2]
    # Final pass
    if zero_counter > 0:
        zeros = str(int(zero_counter / 10)) + str(zero_counter % 10)
        concatenated += zeros
    return concatenated


def decode_number_part(character: str) -> int:
    return [i for i in range(51) if encoding_characters[i] == character][0]


def decode_probabilities(concatenated: str) -> np.ndarray:
    remaining = concatenated
    numbers = []
    while(len(remaining) > 2):
        x, remaining = remaining[0:2], remaining[2:]
        if x[0] in digit_strings:
            for _ in range(int(x)):
                numbers.append(0.0)
        else:
            probability = decode_number_part(x[0]) / 50 + decode_number_part(x[1]) / 2500
            numbers.append(probability)
    # Final pass
    x = remaining
    if x[0] in digit_strings:
        for _ in range(int(x)):
            numbers.append(0.0)
    else:
        probability = decode_number_part(x[0]) / 50 + decode_number_part(x[1]) / 2500
        numbers.append(probability)
    return np.array(numbers)
