num = [2, 4, 5, 6, 7]


def linear_search(number):
    for i, val in enumerate(num):
        if val == number:
            return i


print(linear_search(7))