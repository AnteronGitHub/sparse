import yaml

class Deployment:
    name : str
    dag : dict

    def __init__(self, name : str, dag : dict):
        self.name = name
        self.dag = dag

    def __str__(self):
        return self.name

    @classmethod
    def from_yaml(cls, file_path : str):
        with open(file_path) as f:
            data = yaml.safe_load(f)

        for k in data["dag"].keys(): data["dag"][k] = set(data["dag"][k])

        return cls(data["name"], data["dag"])
