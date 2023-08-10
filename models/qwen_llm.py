from abc import ABC
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Generator
from langchain.callbacks.manager import CallbackManagerForChainRun
from models.loader import LoaderCheckPoint
from models.chatglm_llm import ChatGLMLLMChain
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
# import torch
import transformers


class QWenLLMChain(ChatGLMLLMChain):
    max_token: int = 10000
    temperature: float = 0.01
    # 相关度
    top_p = 0.4
    # 候选词数量
    top_k = 10
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
    def _chain_type(self) -> str:
        return "QWenLLMChain"

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Generator]:
        self.logger.debug(inputs)
        generator = self.generatorAnswer(inputs=inputs, run_manager=run_manager)
        self.logger.debug(generator)
        return {self.output_key: generator}

    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:
        history = inputs[self.history_key]
        streaming = inputs[self.streaming_key]
        prompt = inputs[self.prompt_key]
        self.logger.debug(prompt)
        print(f"__call:{prompt}")
        # Create the StoppingCriteriaList with the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        stopping_criteria_list.append(listenerQueue)
        if streaming:
            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.checkPoint.model.chat_stream(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    # temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stopping_criteria=stopping_criteria_list
            )):
                print(f"_generate_answer->streaming->stream_resp:{stream_resp}")
                # self.checkPoint.clear_torch_cache()
                history[-1] = [prompt, stream_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                generate_with_callback(answer_result)
            self.checkPoint.clear_torch_cache()
        else:
            response, _ = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
                # temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                stopping_criteria=stopping_criteria_list
            )
            print(f"_generate_answer->nostreaming->response:{response}")
            self.checkPoint.clear_torch_cache()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}

            generate_with_callback(answer_result)
