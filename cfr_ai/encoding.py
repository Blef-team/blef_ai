import numpy as np

encoding_characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO'

def encode_probabilities(probabilities: np.ndarray) -> str:
    concatenated = ''
    for p in probabilities:
        inflated = int(p * 2500)
        part_1 = int(inflated / 50)
        part_2 = inflated % 50
        concatenated += encoding_characters[part_1] + encoding_characters[part_2]
    return concatenated

def decode_number_part(character: chr) -> int:
    return [i for i in range(51) if encoding_characters[i] == character][0]

def decode_probabilities(concatenated: str) -> np.ndarray:
    split = [concatenated[i*2:i*2+2] for i in range(int(len(concatenated)/2))]
    numbers = [decode_number_part(x[0]) / 50 + decode_number_part(x[1]) / 2500 for x in split]
    return np.array(numbers)
