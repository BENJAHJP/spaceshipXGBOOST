class Human:
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Person(Human):
    def __init__(self, gender, name, age):
        assert isinstance(gender, str)
        assert isinstance(name, str)
        assert isinstance(age, int)
        self.gender = gender
        super(Person, self).__init__(name=name, age=age)


person1 = Person("male", "B3nah", 56)
print(person1.gender)
print(person1.name)
print(person1.age)
