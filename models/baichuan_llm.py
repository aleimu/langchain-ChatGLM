import traceback
from langchain.callbacks.manager import CallbackManagerForChainRun
from typing import Optional, List, Any, Dict, Generator
from models.loader import LoaderCheckPoint
from models.chatglm_llm import ChatGLMLLMChain
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
from utils.logger import logger


class BaichuanLLMChain(ChatGLMLLMChain):
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

    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:
        try:
            history = inputs[self.history_key]
            streaming = inputs[self.streaming_key]
            prompt = inputs[self.prompt_key]
            messages = []
            messages.append({"role": "user", "content": prompt})
            history = history[-self.history_len:-1] if self.history_len > 0 else []
            if streaming:
                history += [[]]
                for inum, stream_resp in enumerate(self.checkPoint.model.chat(
                        self.checkPoint.tokenizer,
                        messages,
                        stream=True
                )):
                    self.checkPoint.clear_torch_cache()
                    answer_result = AnswerResult()
                    history[-1] = [prompt, stream_resp]
                    answer_result.history = history
                    answer_result.llm_output = {"answer": stream_resp}
                    generate_with_callback(answer_result)
                self.checkPoint.clear_torch_cache()
            else:
                response = self.checkPoint.model.chat(
                    self.checkPoint.tokenizer,
                    messages
                )
                self.checkPoint.clear_torch_cache()
                answer_result = AnswerResult()
                history += [[prompt, response]]
                answer_result.history = history
                answer_result.llm_output = {"answer": response}
                generate_with_callback(answer_result)
            self.checkPoint.clear_torch_cache()
        except Exception:
            traceback.print_exc()
