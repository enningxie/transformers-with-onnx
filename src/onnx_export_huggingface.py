# Created by xieenning at 2020/5/24
from src.convert_graph_to_onnx import convert
from onnxruntime_tools import optimizer

# from os import environ
# from psutil import cpu_count
#
# # Constants from the performance optimization available in onnxruntime
# # It needs to be done before importing onnxruntime
# environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
# environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import time
from onnxruntime import InferenceSession, SessionOptions, get_all_providers
from transformers import BertTokenizerFast, TFBertModel


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties than might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1

    # Load the model as a graph and prepare the CPU backend
    return InferenceSession(model_path, options, providers=[provider])


if __name__ == '__main__':
    MODEL_PATH = '/Data/enningxie/Pretrained_models/transformers_test/bert-base-uncased'
    OUTPUT_PATH = "onnx/bert-base-uncased.onnx"
    OPTIMIZED_MODEL_PATH = "onnx/bert-base-uncased-optimized.onnx"
    total_runs = 100
    # # Step 1: Exporting Huggingface/transformers model to ONNX
    # convert(framework="tf", model=MODEL_PATH, output=OUTPUT_PATH, opset=11)

    # # Step 2(optional): optimizations for bert-base-cased model converted from Tensorflow(tf.keras)
    # optimized_model = optimizer.optimize_model(OUTPUT_PATH, model_type='bert_keras', num_heads=12, hidden_size=768)
    # optimized_model.save_model_to_file(OPTIMIZED_MODEL_PATH)

    # Step 3: Forwarding through our optimized ONNX model running on GPU
    # CUDAExecutionProvider
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = TFBertModel.from_pretrained(MODEL_PATH)

    # Inputs are provided through numpy array
    model_inputs = tokenizer.encode_plus("My name is Bert Bert", return_tensors="tf")

    sequence, pooled = model(model_inputs)
    print(f"Sequence output: {sequence.shape}, Pooled output: {pooled.shape}")

    start_time = time.perf_counter()
    for _ in range(total_runs):
        start_scores, end_scores = model(model_inputs)
    end_time = time.perf_counter()
    print("Tensorflow Inference time {} ms".format(format((end_time - start_time) * 1000 / total_runs, '.2f')))

    unoptimized_model = create_model_for_provider(OUTPUT_PATH, "CUDAExecutionProvider")

    print(f"inputs name: {[tmp_obj.name for tmp_obj in unoptimized_model.get_inputs()]}")
    print(f"inputs shape: {[tmp_obj.shape for tmp_obj in unoptimized_model.get_inputs()]}")

    print(f"outputs name: {[tmp_obj.name for tmp_obj in unoptimized_model.get_outputs()]}")
    print(f"outputs shape: {[tmp_obj.shape for tmp_obj in unoptimized_model.get_outputs()]}")

    inputs_onnx = {k: v.numpy() for k, v in model_inputs.items()}

    # Run the model (None = get all the outputs)
    sequence_01, pooled_01 = unoptimized_model.run(None, inputs_onnx)

    print(f"Sequence output: {sequence_01.shape}, Pooled output: {pooled_01.shape}")

    # measure the latency.
    start = time.perf_counter()
    for _ in range(total_runs):
        opt_results = unoptimized_model.run(None, inputs_onnx)
    end = time.perf_counter()
    print("ONNX Runtime gpu inference time on unoptimized model: {} ms".format(
        format((end - start) * 1000 / total_runs, '.2f')))
    del unoptimized_model

    print('-----------------------------------')
    optimized_model = create_model_for_provider(OPTIMIZED_MODEL_PATH, "CUDAExecutionProvider")

    print(f"inputs name: {[tmp_obj.name for tmp_obj in optimized_model.get_inputs()]}")
    print(f"inputs shape: {[tmp_obj.shape for tmp_obj in optimized_model.get_inputs()]}")

    print(f"outputs name: {[tmp_obj.name for tmp_obj in optimized_model.get_outputs()]}")
    print(f"outputs shape: {[tmp_obj.shape for tmp_obj in optimized_model.get_outputs()]}")

    # Run the model (None = get all the outputs)
    sequence_02, pooled_02 = optimized_model.run(None, inputs_onnx)

    print(f"Sequence output: {sequence_02.shape}, Pooled output: {pooled_02.shape}")

    # measure the latency.
    start = time.perf_counter()
    for _ in range(total_runs):
        opt_results = optimized_model.run(None, inputs_onnx)
    end = time.perf_counter()
    print("ONNX Runtime gpu inference time on optimized model: {} ms".format(
        format((end - start) * 1000 / total_runs, '.2f')))
    del optimized_model
