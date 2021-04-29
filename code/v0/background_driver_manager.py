from enum import Enum
from pathlib import Path
from typing import List, Optional
from concurrent.futures import Future, ThreadPoolExecutor
from traceback import format_exc
from aiochannel import Channel
from skimage.io import imread, imsave
from traitlets import HasTraits, default
from traitlets import List as TList
from threading import Thread
from ipywidgets import Output
import tempfile

from output_logger import getLogger, debug_capture
from digitize_analysis import DigitizeImageAnalysis
from analysis_driver import AnalysisDriver
from base_analysis import VarintMode
from async_utils import IExecutorFactory


class RunState(Enum):
    NOT_RUNNING = "not_running",
    STARTING = "starting",
    RUNNING = "running",
    STOPPING = "stopping";
    

class ShuttingDownException(RuntimeError):
    pass

LOGGER = getLogger(__name__)

class BackgroundDriverManager(HasTraits):
    result_data = TList()
    
    def __init__(self, executor_factory: IExecutorFactory, thread_count=1):
        self._executor_factory: IExecutorFactory = executor_factory
        self._work_executor: Optional[ThreadPoolExecutor] = None
        self._thread_count: int = thread_count
        self._status: RunState = RunState.NOT_RUNNING
        
    @default('result_data')
    def default_result_data() -> List:
        return []
    
    # @debug_capture()
    async def start(self) -> None:
        if self._status == RunState.NOT_RUNNING:
            self._status = RunState.STARTING
            LOGGER.info("Running BDM::start()")
            self._work_executor = self._executor_factory.create_executor(
                max_workers=self._thread_count, thread_name_prefix="storage-analysis-worker-")
            self._status = RunState.RUNNING
            LOGGER.info("BDM is RUNNNING")
        elif self._status != RunState.RUNNING:
            raise RuntimeError(f"Cannot start from state = {self._status}")

    # @debug_capture()
    async def stop(self) -> None:
        if self._status == RunState.RUNNING:
            self._status = RunState.STOPPING
            with self._debug_output:
                LOGGER.info("Running BDM::stop()")
            await self._work_executor.shutdown(wait=True)
            self._status = RunState.NOT_RUNNING
        elif self._status != RunState.NOT_RUNNING:
            raise RuntimeError(f"Cannot stop from state = {self._status}")

    def _in_background(self, analysis_task: AnalysisDriver) -> None:
        try:
            LOGGER.info("Running BDM::in_background()")
            work_remains = True
            while work_remains and self._status == RunState.RUNNING:
                LOGGER.info("Entering next analysis step...")
                work_remains = analysis_task.run_analysis()
                LOGGER.info("Analysis step completed")
            LOGGER.info("Analysis complete, returning thread to pool")
        except Exception as e:
            LOGGER.error(format_exc())
            raise e

    # @debug_capture()
    def submit_task(self, analysis_task: AnalysisDriver) -> Future:
        LOGGER.info("Running BDM::submit_task()")
        if self._status == RunState.RUNNING:
            try:
                return self._work_executor.submit(self._in_background, analysis_task)
            except Exception as e:
                LOGGER.error(format_exc())
                raise e
        else:
            LOGGER.info(f"BDM is {self._status}")
            raise ShuttingDownException()
