class Human:
    __name = ""
    __age = 0

    def set_name(self, name):
        self.__name = name

    @property
    def get_name(self):
        return self.__name

    def set_age(self, age):
        assert isinstance(age, int)
        self.__age = age

    @property
    def get_age(self):
        return self.__age


human = Human()
human.set_name("name")
print(human.get_name)
human.set_age(67)
print(human.get_age)
