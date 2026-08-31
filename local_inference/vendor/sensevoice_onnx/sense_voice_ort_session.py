# -*- coding:utf-8 -*-
# Based on lovemefan/SenseVoice-python (MIT); OrtInferRuntimeSession extended for DirectML.
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import sentencepiece as spm
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
)

logger = logging.getLogger(__name__)


class OrtInferRuntimeSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        dml_ep = "DmlExecutionProvider"
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        available = get_available_providers()
        EP_list = []
        want_gpu = device_id != "-1"
        if want_gpu:
            if cuda_ep in available:
                EP_list.append((cuda_ep, cuda_provider_options))
            elif dml_ep in available:
                EP_list.append((dml_ep, {}))
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(model_file)

        self.session = InferenceSession(
            model_file, sess_options=sess_opt, providers=EP_list
        )

        if want_gpu and cuda_ep not in self.session.get_providers() and dml_ep not in self.session.get_providers():
            warnings.warn(
                f"GPU execution was requested but neither {cuda_ep} nor {dml_ep} is active; "
                f"encoder runs on {cpu_ep}. For CUDA, match onnxruntime-gpu with your CUDA/cuDNN. "
                "For Windows integrated GPU, install onnxruntime-directml.",
                RuntimeWarning,
                stacklevel=2,
            )

    def __call__(self, input_content) -> list:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as exc:
            logger.exception("ONNX Runtime inference failed")
            raise RuntimeError("ONNX Runtime inference failed") from exc

    def get_input_names(self):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self):
        return [v.name for v in self.session.get_outputs()]

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


class SenseVoiceInferenceSession:
    def __init__(
        self,
        embedding_model_file,
        encoder_model_file,
        bpe_model_file,
        device_id=-1,
        intra_op_num_threads=4,
    ):
        logger.info("Loading SenseVoice embeddings from %s", embedding_model_file)

        self.embedding = np.load(embedding_model_file)
        logger.info("Loading SenseVoice encoder %s", encoder_model_file)
        start = time.time()
        self.encoder = OrtInferRuntimeSession(
            encoder_model_file,
            device_id=device_id,
            intra_op_num_threads=intra_op_num_threads,
        )
        logger.info("Loading encoder took %.2f s", time.time() - start)
        self.blank_id = 0
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_file)

    def __call__(self, speech, language: int, use_itn: bool) -> str:
        language_query = self.embedding[[[language]]]

        text_norm_query = self.embedding[[[14 if use_itn else 15]]]
        event_emo_query = self.embedding[[[1, 2]]]

        input_content = np.concatenate(
            [
                language_query,
                event_emo_query,
                text_norm_query,
                speech,
            ],
            axis=1,
        ).astype(np.float32)
        input_length = np.array([input_content.shape[1]], dtype=np.int64)

        encoder_out = self.encoder((input_content, input_length))[0]

        def unique_consecutive(arr):
            if len(arr) == 0:
                return arr
            mask = np.append([True], arr[1:] != arr[:-1])
            out = arr[mask]
            out = out[out != self.blank_id]
            return out.tolist()

        hypos = unique_consecutive(encoder_out[0].argmax(axis=-1))
        return self.sp.DecodeIds(hypos)
