class TaskExecutor:
    """Common base class for task execution logic. This class implements potentially hardware-accelerated computations
    that are offloaded into worker nodes.

    User is expected to implement the computation logic by defining a custom execute_task() function. Additionally it
    is possible to implement custom initialization code by overriding optional start() hook.
    """
    def start():
        pass

    def execute_task(input_data : bytes) -> bytes:
        raise "Task executor not implemented! See documentation on how to implement your own executor"

