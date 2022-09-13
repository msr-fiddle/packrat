import pickle
from pyarrow import plasma
import pyarrow as pa

# PyTorch model
import torch
from torchvision import models
import transformers

from .serde import extract_tensors, pack_frames, replace_tensors, unpack_frames


def get_model_from_torch(name: str) -> torch.nn.Module:

    if name == "resnet50":
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet50.eval()
        return resnet50

    if name == "inception":
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()
        return inception

    if name == "bert":
        transformers.logging.set_verbosity_error()
        bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        bert.eval()
        return bert

    if name == "gpt2":
        gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2.eval()
        return gpt2

    raise Exception(f"Unknown model: {name}")


def get_model_from_plasma(storename: str, model_name: str) -> torch.nn.Module:
    client = plasma.connect(storename)
    metadata = pickle.loads(client.get(plasma.ObjectID(2 * b"metadata00")))
    serial_ref = metadata[model_name]["serial"]
    buf_ref = metadata[model_name]["buf_id"]
    (model, weights) = pickle.loads(client.get(serial_ref),
                                    buffers=unpack_frames(client.get(buf_ref)))
    replace_tensors(model, weights)
    model.eval()
    client.disconnect()
    return model


class Cache:
    """
    Zero-copy model transfer from the cache to the client
    """

    def __init__(self):
        """
        Cache for storing models and their weights
        params:
            storename: name of the plasma store
            benchmark: name of the benchmark
            buf_id: id of the out-of-band buffer
            serial: id of the serialized model
        """
        self.store = plasma.start_plasma_store(1 * 1024 * 1024 * 1024)
        self.storename, _ignore = self.store.__enter__()
        self.client = plasma.connect(self.storename)

        self.metadata = {}

        benchmarks = ["resnet50"]
        for self.benchmark in benchmarks:
            ouf_of_band_buffer = []
            model = get_model_from_torch(self.benchmark)
            serialized_model = pickle.dumps(
                extract_tensors(model),
                buffer_callback=lambda b: ouf_of_band_buffer.append(b.raw()),
                protocol=pickle.HIGHEST_PROTOCOL
            )

            _sz, byte_ls = pack_frames(ouf_of_band_buffer)
            buf = pa.py_buffer(b"".join(byte_ls))

            buf_id = self.client.put(buf)
            serial = self.client.put(serialized_model)

            self.metadata[self.benchmark] = {
                "buf_id": buf_id,
                "serial": serial
            }

            # TODO: Figure out what to do with the buffers (fixme)
            del ouf_of_band_buffer
            del model
            del serialized_model
            del buf
            del byte_ls

        self.client.put(pickle.dumps(self.metadata),
                        object_id=plasma.ObjectID(2 * b"metadata00"))

    def __str__(self):
        return self.__class__.__name__

    def get_model_from_cache(self, benchmark: str) -> torch.nn.Module:
        metadata = pickle.loads(self.client.get(
            plasma.ObjectID(2 * b"metadata00")))
        serial_ref = metadata[benchmark]["serial"]
        buf_ref = metadata[benchmark]["buf_id"]
        (model, weights) = pickle.loads(self.client.get(serial_ref),
                                        buffers=unpack_frames(self.client.get(buf_ref)))
        replace_tensors(model, weights)
        return model


if __name__ == '__main__':
    cache = Cache()
    resnet = cache.get_model_from_cache("resnet50")
    resnet.eval()
