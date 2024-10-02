class Deployment:
    name : str
    dag : dict

    def __init__(self, name : str, dag : dict):
        self.name = name
        self.dag = dag

    def __str__(self):
        return self.name
