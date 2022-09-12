import os
import pickle
import unittest
import warnings
import transformers
import torch
import torchvision

from psutil import Process
from pyarrow import plasma
from benchmarks.cache.serde import replace_tensors, unpack_frames
from benchmarks.cache.store import Cache, get_model


class TestCache(unittest.TestCase):
    """
    Testcases for the cache
    """

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=UserWarning)

    def test_plama_connect(self) -> None:
        cache = Cache()
        self.assertIsNotNone(cache.storename)
        self.assertIsNotNone(cache.serial)
        self.assertIsNotNone(cache.buf_id)

        client = plasma.connect(cache.storename)
        self.assertIsNotNone(client)
        self.assertIsNotNone(client.list())
        self.assertEqual(client.store_capacity(), 1*1024*1024*1024)
        client.disconnect()

    def test_plasma_zero_copy(self) -> None:
        process = Process(os.getpid())

        cache = Cache()
        self.assertIsNotNone(cache.storename)
        self.assertIsNotNone(cache.serial)
        self.assertIsNotNone(cache.buf_id)
        client = plasma.connect(cache.storename)

        memory_info_before = process.memory_info().rss

        _objects = []
        for _i in range(100):
            buffer = client.get(cache.buf_id)
            serialized = client.get(cache.serial)
            _objects.append((buffer, serialized))

        memory_info_after = process.memory_info().rss
        self.assertEqual(memory_info_before, memory_info_after)
        client.disconnect()

    def test_plasma_zero_copy_with_model(self) -> None:
        process = Process(os.getpid())

        cache = Cache()
        self.assertIsNotNone(cache.storename)
        self.assertIsNotNone(cache.serial)
        self.assertIsNotNone(cache.buf_id)
        client = plasma.connect(cache.storename)

        memory_info_before = process.memory_info().rss

        model_count = 100
        max_allowed_mem = model_count * 1 * 1e6  # Each model reconstructs to max 1MB

        _objects = []
        for _i in range(model_count):
            (model, weights) = pickle.loads(client.get(cache.serial),
                                            buffers=unpack_frames(client.get(cache.buf_id)))
            replace_tensors(model, weights)
            model.eval()
            _objects.append(model)

        memory_info_after = process.memory_info().rss
        self.assertTrue(memory_info_before +
                        max_allowed_mem >= memory_info_after)
        client.disconnect()

    def test_model_isinstance_resnet(self) -> None:
        cache = Cache()
        resnet = cache.get_model()
        resnet.eval()

        self.assertIsInstance(resnet, torch.nn.Module)
        self.assertIsInstance(resnet, torchvision.models.resnet.ResNet)
        self.assertNotIsInstance(
            resnet, torchvision.models.inception.Inception3)

    def test_get_model(self) -> None:
        benchmarks = ["resnet50", "inception", "bert", "gpt2"]
        for benchmark in benchmarks:
            model = get_model(benchmark)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, torch.nn.Module)

            # Check if the model is an instance of the correct model
            if benchmark == "resnet50":
                self.assertIsInstance(model, torchvision.models.resnet.ResNet)
            if benchmark == "inception":
                self.assertIsInstance(
                    model, torchvision.models.inception.Inception3)
            if benchmark == "bert":
                self.assertIsInstance(
                    model, transformers.models.bert.modeling_bert.BertModel)
            if benchmark == "gpt2":
                self.assertIsInstance(
                    model, transformers.models.gpt2.modeling_gpt2.GPT2Model)
