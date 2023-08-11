import traceback
from abc import ABC
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForChainRun
from typing import Optional, List, Any, Dict, Generator
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
from utils.logger import logger


class BaichuanLLMChain(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10
    streaming_key: str = "streaming"  #: :meta private:
    history_key: str = "history"  #: :meta private:
    prompt_key: str = "prompt"  #: :meta private:
    output_key: str = "answer_result_stream"  #: :meta private:

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "BaichuanLLMChain"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
            self,
            inputs: Dict[str, Any],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Generator]:
        logger.debug(inputs)
        generator = self.generatorAnswer(inputs=inputs, run_manager=run_manager)
        logger.debug(f"__call->generator:{generator}")
        return {self.output_key: generator}

    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:
        try:
            history = inputs[self.history_key]
            streaming = inputs[self.streaming_key]
            prompt = inputs[self.prompt_key]
            logger.debug(f"__call->_generate_answer:{prompt}")
            messages = []
            messages.append({"role": "user", "content": prompt})
            if streaming:
                for inum, stream_resp in enumerate(self.checkPoint.model.chat(
                        self.checkPoint.tokenizer,
                        messages,
                        stream=True
                )):
                    logger.debug(f"_generate_answer->streaming->response:{stream_resp}")
                    self.checkPoint.clear_torch_cache()
                    answer_result = AnswerResult()
                    answer_result.llm_output = {"answer": stream_resp}
                    yield answer_result
            else:
                response = self.checkPoint.model.chat(
                    self.checkPoint.tokenizer,
                    messages
                )
                logger.debug(f"_generate_answer->nostreaming->response:{response}")
                self.checkPoint.clear_torch_cache()
                answer_result = AnswerResult()
                answer_result.llm_output = {"answer": response}
                yield answer_result
        except Exception:
            traceback.print_exc()
