import asyncio
import tempfile
import traceback
from enum import Enum
from pathlib import Path
from aiochannel import Channel

from skimage.io import imread, imsave
from traitlets import HasTraits, Int, Unicode, Set, Instance

from digitize_analysis import DigitizeImageAnalysis
from base_analysis import BaseAnalysis, VarintMode
from output_logger import getLogger


LOGGER = getLogger(__name__)

class AnalysisDriver():
    completed_experiments = Set(
        default_value = set(), 
        trait = Instance(klass = VarintMode))
    def __init__(
        self, source_file_name: str, source_file_data: bytes, temp_dir: Path,
        result_channel: Channel, loop: asyncio.AbstractEventLoop, enable_varint: bool = False
    ):
        temp_root_path = Path(temp_dir)
        self._source_file_name = source_file_name
        self._enable_varint: bool = enable_varint
        self._loop: asyncio.AbstractEventLoop = loop
        self._result_channel: Channel = result_channel
        self._temp_dir: tempfile.TemporaryDirectory = temp_dir
        self._image_analysis: Optional[DigitizeImageAnalysis] = DigitizeImageAnalysis(
            source_file_name, source_file_data, str(temp_root_path / "analysis-result_{}.{}")
        )
        self._metric_count = 3 + (3 * 6) if enable_varint else 3 + (3 * 4)
        
        # Varint and Sparse variants need output of from initial baseline analysis to
        # define themselves, so we defer their construction until during analysis.
        self._additional_steps = None
        self._dense_varint_image_analysis = None
        self._sparse_varint_image_analysis = None
        self._dense_proto_image_analysis = None
        self._sparse_proto_image_analysis = None
        self._dense_rawbin_image_analysis = None
        self._sparse_rawbin_image_analysis = None
        
    def run_bootstrap_analysis(self) -> bool:
        if self._image_analysis is None or not self._additional_steps is None:
            raise RuntimeException("Invalid state for initial bootstrap run")
        LOGGER.info("Performing initial analysis step")
        self._image_analysis.run_analysis()
#         LOGGER.info("Reporting initial analysis measurements")
#         asyncio.run_coroutine_threadsafe(
#             self._report_results(self._image_analysis), self._loop)
        LOGGER.info("Allocating density and varint experiments")
        self._dense_proto_image_analysis = self._image_analysis.to_varint_sparse_analysis(
            VarintMode.USE_PROTO_INT, False)
        self._sparse_proto_image_analysis = self._image_analysis.to_varint_sparse_analysis(
            VarintMode.USE_PROTO_INT, True)
        self._dense_varint_image_analysis = self._image_analysis.to_varint_sparse_analysis(
            VarintMode.USE_PY_VARINT, False)
        self._sparse_varint_image_analysis = self._image_analysis.to_varint_sparse_analysis(
            VarintMode.USE_PY_VARINT, True)
        self._dense_rawbin_image_analysis = self._image_analysis.to_varint_sparse_analysis(
            VarintMode.USE_BYTES, False)
        self._sparse_rawbin_image_analysis = self._image_analysis.to_varint_sparse_analysis(
            VarintMode.USE_BYTES, True)
        self._additional_steps = [
            self._dense_proto_image_analysis, self._sparse_proto_image_analysis,
            self._dense_varint_image_analysis, self._sparse_varint_image_analysis,
            self._dense_rawbin_image_analysis, self._sparse_rawbin_image_analysis
        ] if self._enable_varint else [
            self._dense_proto_image_analysis, self._sparse_proto_image_analysis,
            self._dense_rawbin_image_analysis, self._sparse_rawbin_image_analysis
        ]
        LOGGER.info("Reporting initial analysis measurements")
        
        asyncio.run_coroutine_threadsafe(
            self._report_results(self._image_analysis), self._loop)
        
    def get_metric_count(self) -> int:
        # TODO: If digitize is byte sparing, we skip the varint variants and metrics will overcount by four,
        #       from two skipped sparse runs and two skipped dense runs.  We cannot discover this until after
        #       the initial analysis has been run, but by then the progress meter is already allocated...
        return self._metric_count
    
    def run_analysis(self) -> bool:
        try:
            if self._image_analysis is None:
                LOGGER.info("Post-comp1letion call to run_analysis()")
                return False
            else:
                next_step: BaseAnalysis = self._additional_steps.pop()
                LOGGER.info("Performing next additional analysis step")
                next_step.run_analysis()
                LOGGER.info("Reporting next analysis measurements")
                asyncio.run_coroutine_threadsafe(
                    self._report_results(next_step), self._loop)
            
            if not self._aditional_steps is None and len(self._additional_steps) == 0:
                self._additional_steps = None
                self._image_analysis = None
                LOGGER.info("Returning False to indicate the work is done!")
                return False
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            raise e
        LOGGER.info("Returning True to continue the work!")
        return True
    
    async def _report_results(self, analysis: BaseAnalysis):
        results = []
        analysis.get_results(results)
        for result in results:
            await self._result_channel.put(result)
        if not self._aditional_steps is None and len(self._additional_steps) == 0:
            await self._result_channel.put({
                "msg_type": "analysis_complete",
                "source": self._source_file_name
            })
