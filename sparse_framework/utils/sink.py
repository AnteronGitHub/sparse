class SparseSink:
    @property
    def name(self):
        return self.__class__.__name__

    def tuple_received(self, new_tuple):
        pass

