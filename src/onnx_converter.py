# Created by xieenning at 2020/5/26
"""ONNX model converter."""
import os
from keras2onnx import convert_keras, save_model
from onnxruntime_tools import optimizer
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
from transformers import BertConfig, BertTokenizer, TFBertModel


class ONNXConverterTF(object):
    def __init__(self, tokenizer, num_heads=12, hidden_size=768, target_opset=12):
        """
        :param tokenizer: Target tokenizer with the model.
        :param num_heads: Number of attention heads for each attention layer in the Transformer encoder.
        :param hidden_size: Dimensionality of the encoder layers and the pooler layer.
        :param target_opset: The targeted onnx model opset; Opset has been updated to version 12 (ONNX 1.7.0).
        """
        self.tokenizer = tokenizer
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.target_opset = target_opset
        self.sample_inputs = None

    def convert(self, model, saved_path, model_name):
        """
        Convert bert model (transformers) to onnx optimized model.
        :param model: Trained model from transformers.
        :param saved_path: The path to save onnx model.
        :param model_name: Choose a model name to save.
        :returns
            optimized_model: Optimized onnx model.
            optimized_model_saved_path: optimized model saved path.
        """
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        unoptimized_model_saved_path = os.path.join(saved_path, '{}.onnx'.format(model_name))
        optimized_model_saved_path = os.path.join(saved_path, '{}_optimized.onnx'.format(model_name))
        self.sample_inputs = self.tokenizer.encode_plus("This is a sample input", return_tensors='tf')
        # Step 1: Convert origin transformers model to unoptimized ONNX model.
        model.predict(self.sample_inputs.data)
        unoptimized_model = convert_keras(model, model.name, target_opset=self.target_opset)
        save_model(unoptimized_model, unoptimized_model_saved_path)

        # Step 2: optimizations for trained model converted from Tensorflow(tf.keras)
        optimized_model = optimizer.optimize_model(unoptimized_model_saved_path, model_type='bert_keras',
                                                   num_heads=self.num_heads, hidden_size=self.hidden_size)
        optimized_model.save_model_to_file(optimized_model_saved_path)

        return optimized_model, optimized_model_saved_path

    def verify(self, onnx_model_path):
        """
        :param onnx_model_path: onnx model path.
        """
        print("Checking ONNX model loading from: {}".format(onnx_model_path))
        try:
            onnx_options = SessionOptions()
            sess = InferenceSession(onnx_model_path, onnx_options, providers=["CUDAExecutionProvider"])
            print("Model correctly loaded")
            if self.sample_inputs is not None:
                inputs_onnx = {k: v.numpy() for k, v in self.sample_inputs.items()}
                print(f"Model inputs name: {[tmp_obj.name for tmp_obj in sess.get_inputs()]}")
                print(f"Model inputs shape: {[tmp_obj.shape for tmp_obj in sess.get_inputs()]}")
                print(f"Model outputs name: {[tmp_obj.name for tmp_obj in sess.get_outputs()]}")
                print(f"Model outputs shape: {[tmp_obj.shape for tmp_obj in sess.get_outputs()]}")
                # Run the model (None = get all the outputs)
                outputs_onnx = sess.run(None, inputs_onnx)
                print("Model inference correctly")
        except RuntimeException as re:
            print("Error while loading the model: {}".format(re))


if __name__ == '__main__':
    # Example
    model_name_or_path = "bert-base-uncased"
    # load origin model from transformers.
    tmp_config = BertConfig.from_pretrained(model_name_or_path)
    tmp_tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    tmp_model = TFBertModel(tmp_config).from_pretrained(model_name_or_path)

    # convert
    tmp_saved_path = '/tmp/onnx_saved_model'
    tmp_onnx_converter = ONNXConverterTF(tmp_tokenizer)
    _, optimized_model_saved_path = tmp_onnx_converter.convert(tmp_model, tmp_saved_path, 'tf_bert_model')
    tmp_onnx_converter.verify(optimized_model_saved_path)
