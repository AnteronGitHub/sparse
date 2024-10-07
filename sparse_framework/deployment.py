import yaml

class Deployment:
    name : str
    streams : list
    pipelines : dict

    def __init__(self, name : str, streams : set, pipelines : dict):
        self.name = name
        self.streams = streams
        self.pipelines = pipelines

    def __str__(self):
        return self.name

    @classmethod
    def from_yaml(cls, file_path : str):
        with open(file_path) as f:
            data = yaml.safe_load(f)

        return cls(data["name"], data["streams"], data["pipelines"])
