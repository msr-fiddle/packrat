import os
import pickle
import unittest
import warnings
import transformers
import torch
import torchvision

from psutil import Process
from pyarrow import plasma
from benchmarks.cache.store import Cache, get_model_from_plasma, get_model_from_torch
from benchmarks.config import Benchmark


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
                return False
            else:
                raise Exception
    if models_differ == 0:
        return True


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
        self.assertIsNotNone(cache.metadata)

        client = plasma.connect(cache.storename)
        self.assertIsNotNone(client)
        self.assertIsNotNone(client.list())
        self.assertEqual(client.store_capacity(), 1*1024*1024*1024)
        client.disconnect()

    def test_plasma_zero_copy(self) -> None:
        process = Process(os.getpid())

        cache = Cache()
        self.assertIsNotNone(cache.storename)
        self.assertIsNotNone(cache.metadata)

        client = plasma.connect(cache.storename)

        memory_info_before = process.memory_info().rss
        metadata = pickle.loads(client.get(plasma.ObjectID(2 * b"metadata00")))

        _objects = []
        for _i in range(100):
            buffer = client.get(metadata[Benchmark.resnet.name]["buf_id"])
            serialized = client.get(metadata[Benchmark.resnet.name]["serial"])
            _objects.append((buffer, serialized))

        memory_info_after = process.memory_info().rss
        self.assertEqual(memory_info_before, memory_info_after)
        client.disconnect()

    def test_zero_copy_model_from_plasma(self) -> None:
        process = Process(os.getpid())

        cache = Cache()
        self.assertIsNotNone(cache.storename)
        self.assertIsNotNone(cache.metadata)

        memory_info_before = process.memory_info().rss

        model_count = 100
        max_allowed_mem = model_count * 1 * 1e6  # Each model reconstructs to max 1MB

        _objects = []
        for _i in range(model_count):
            model = get_model_from_plasma(
                cache.storename, Benchmark.resnet.name)
            self.assertIsInstance(model, torchvision.models.resnet.ResNet)
            _objects.append(model)

        memory_info_after = process.memory_info().rss
        self.assertTrue(memory_info_before +
                        max_allowed_mem >= memory_info_after)

    def test_zero_copy_model_from_cache(self) -> None:
        process = Process(os.getpid())

        cache = Cache()
        self.assertIsNotNone(cache.storename)
        self.assertIsNotNone(cache.metadata)

        memory_info_before = process.memory_info().rss

        model_count = 100
        max_allowed_mem = model_count * 1 * 1e6  # Each model reconstructs to max 1MB

        _objects = []
        for _i in range(model_count):
            model = cache.get_model_from_cache(Benchmark.resnet.name)
            self.assertIsInstance(model, torchvision.models.resnet.ResNet)
            _objects.append(model)

        memory_info_after = process.memory_info().rss
        self.assertTrue(memory_info_before +
                        max_allowed_mem >= memory_info_after)

    def test_model_isinstance_resnet(self) -> None:
        cache = Cache()
        resnet = cache.get_model_from_cache(Benchmark.resnet.name)
        resnet.eval()

        self.assertIsInstance(resnet, torch.nn.Module)
        self.assertIsInstance(resnet, torchvision.models.resnet.ResNet)
        self.assertNotIsInstance(
            resnet, torchvision.models.inception.Inception3)

    def test_get_model(self) -> None:
        benchmarks = [bench.name for bench in Benchmark]
        for benchmark in benchmarks:
            model = get_model_from_torch(benchmark)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, torch.nn.Module)

            # Check if the model is an instance of the correct model
            if benchmark == Benchmark.resnet.name:
                self.assertIsInstance(model, torchvision.models.resnet.ResNet)
            if benchmark == Benchmark.inception.name:
                self.assertIsInstance(
                    model, torchvision.models.inception.Inception3)
            if benchmark == Benchmark.bert.name:
                self.assertIsInstance(
                    model, transformers.models.bert.modeling_bert.BertModel)
            if benchmark == Benchmark.gpt2.name:
                self.assertIsInstance(
                    model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel)

    def test_plasma_storename_length(self) -> None:
        """
        Test length of the plasma store name does not vary across runs.
        Passing variable length store names to the config file causes
        the benchmark to fail.
        Naming convention: /tmp/test_plasma-<random 8 chars>/plasma.sock
        https://github.com/apache/arrow/blob/b789226ccb2124285792107c758bb3b40b3d082a/python/pyarrow/plasma.py#L105
        """
        for _i in range(10):
            store = plasma.start_plasma_store(1 * 1024 * 1024)
            storename, _ignore = store.__enter__()
            self.assertEqual(len(storename), len(
                "/tmp/test_plasma-xxxxxxxx/plasma.sock"))
            del store

    def test_model_weights_equality(self) -> None:
        torch_resnet = get_model_from_torch(Benchmark.resnet.name)
        torch_resnet.eval()

        cache = Cache()
        resnet_cache = cache.get_model_from_cache(Benchmark.resnet.name)
        resnet_cache.eval()
        compare_models(torch_resnet, resnet_cache)

    def test_model_weights_equality_across_inferences(self) -> None:
        torch_resnet = get_model_from_torch(Benchmark.resnet.name)
        torch_resnet.eval()

        cache = Cache()
        resnet_cache = cache.get_model_from_cache(Benchmark.resnet.name)
        resnet_cache.eval()

        resnet_cache(torch.randn(1, 3, 224, 224))
        compare_models(torch_resnet, resnet_cache)
