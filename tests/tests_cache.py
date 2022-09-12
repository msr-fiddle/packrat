import os
import pickle
import unittest
import warnings

from psutil import Process
from pyarrow import plasma
from benchmarks.cache.serde import replace_tensors, unpack_frames
from benchmarks.cache.store import Cache


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
        plasma.connect(cache.storename)

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

    def test_plasma_zero_copy_with_model(self) -> None:
        process = Process(os.getpid())

        cache = Cache()
        self.assertIsNotNone(cache.storename)
        self.assertIsNotNone(cache.serial)
        self.assertIsNotNone(cache.buf_id)
        client = plasma.connect(cache.storename)

        memory_info_before = process.memory_info().rss

        _objects = []
        for _i in range(100):
            (model, weights) = pickle.loads(client.get(cache.serial),
                                            buffers=unpack_frames(client.get(cache.buf_id)))
        replace_tensors(model, weights)
        model.eval()
        _objects.append(model)

        memory_info_after = process.memory_info().rss
        self.assertEqual(memory_info_before, memory_info_after)
