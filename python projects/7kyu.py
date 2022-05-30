def filter_strings(l):
    for values in l:
        if isinstance(values, str) is False:
            print(values)


print(filter_strings([3, 5, 6, "67", "56"]))
