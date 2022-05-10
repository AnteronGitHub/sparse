import asyncio

from .master import TaskDeployer

class TaskDeployerLegacy(TaskDeployer):
    """Legacy asyncio implementation for older Python compiler versions.
    """

    @asyncio.coroutine
    def stream_task_synchronous(self, input_data : bytes, loop):
        reader, writer = yield from asyncio.open_connection(self.upstream_host, self.upstream_port, loop=loop)

        writer.write(input_data)
        writer.write_eof()

        result_data = yield from reader.read()
        writer.close()

        self.logger.debug("Received result for offloaded task")

        return result_data

    def deploy_task(self, input_data : bytes):
        loop = asyncio.get_event_loop()
        result_data = loop.run_until_complete(self.stream_task_synchronous(input_data, loop))
        # loop.close()
        return result_data
