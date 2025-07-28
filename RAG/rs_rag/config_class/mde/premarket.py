import yaml


class Config:
    def __new__(self, file=None):
        if file is None:
            raise AttributeError(
                "A YAML config file (bytestream or filename) was not passed"
            )
        elif type(file) == bytes:
            self.stream = file
        else:
            self.stream = open(str(file), "rb").read()
        result = yaml.load(self.stream, Loader=yaml.BaseLoader)

        return result
