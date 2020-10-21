from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from aiochannel import Channel
from skimage.io import imread
import numpy as np
import tempfile


def wait_for_change(widget, value):
    future = asyncio.Future()
    
    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)
    widget.observe(getvalue, value)
    
    return future

def png_bytes_to_ndarray(raw_png_bytes: bytes) -> np.ndarray:
    upload_content = upload_record[key]["content"]
    with open("temp.png", "wb") as foo:
        foo.write(upload_content)
    # TODO: tempfile = tempfile.
    return imread("temp.png")
    
    
class IChannelFactory(ABC):
    @abstractmethod
    def create_channel(self, buffer_size: int = 1) -> Channel:
        pass
    
    
class IExecutorFactory(ABC):
    @abstractmethod
    def create_executor(self, max_workers: Optional[int] = None, thread_name_prefix: Optional[str] = None) -> ThreadPoolExecutor:
        pass
    
    
class ChannelFactory(IChannelFactory):
    def create_channel(self, buffer_size: int = 1) -> Channel:
        """
        This factory function exists so components may request channels with specific buffer capacities while still
        acquiriing them through a Dependency Injection technique that can be replaced with a mock implementation if
        required for testing purposes.
        """
        return Channel(buffer_size)
    
    
class ExecutorFactory(IExecutorFactory):
    def create_executor(self, max_workers: Optional[int] = None, thread_name_prefix: Optional[str] = None) -> ThreadPoolExecutor:
        """
        This factory function exists so components may request channels with specific buffer capacities while still
        acquiriing them through a Dependency Injection technique that can be replaced with a mock implementation if
        required for testing purposes.
        """
        return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
    
    
# This symbol only intended as a convenience for bootstrapping.  Any routine besides main() is intended to 
# acquire an IChanneFactory by DI only for the sake of testability.
CHANNEL_FACTORY_SINGLETON: IChannelFactory = ChannelFactory()
EXECUTOR_FACTORY_SINGLETON: IExecutorFactory = ExecutorFactory()