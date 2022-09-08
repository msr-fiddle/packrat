import gc
from itertools import count
import struct
import torch
from typing import *
import copy


def test_plasma_with_numpy_array():
    """Benchmark plasma store with numpy arrays."""

    from psutil import Process
    import os
    import numpy as np
    from pyarrow import plasma

    p = Process(os.getpid())
    store = plasma.start_plasma_store(1024 * 1024 * 1024)
    client = plasma.connect(store.__enter__()[0])
    serialized_model = np.array([i for i in range(100000000)])
    id = client.put(serialized_model)

    # Take readings after putting the object in the store
    memory_info_before = p.memory_info().rss

    # Make 100 copies of the object
    objects = []
    for i in range(100):
        objects.append(client.get(id))

    # Take readings after making 100 copies of the object
    memory_info_after = p.memory_info().rss

    # Verify that the objects are the same
    assert memory_info_before == memory_info_after
    print(f"Used Memory {memory_info_after / 1e9} GB")


def test_pickled_numpy_array():
    """Benchmark plasma store with pickled numpy arrays."""

    from psutil import Process
    import os
    import numpy as np
    import pickle

    buffer = []
    p = Process(os.getpid())
    serialized_model = pickle.dumps(
        np.array([i for i in range(100000000)]), buffer_callback=buffer.append, protocol=pickle.HIGHEST_PROTOCOL)

    # Take readings after putting the object in the store
    memory_info_before = p.memory_info().rss

    # Make 100 copies of the object
    objects = []
    for _i in range(100):
        objects.append(pickle.loads(serialized_model, buffers=buffer))

    # Take readings after making 100 copies of the object
    memory_info_after = p.memory_info().rss

    # Verify that the objects are the same
    assert memory_info_before == memory_info_after
    print(f"Used Memory {memory_info_after / 1e9} GB")


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})

    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in ([name for name, _ in module.named_parameters(recurse=False)]
                     + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)

    # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict]):
    """
    Restore the tensors that extract_tensors() stripped out of a
    PyTorch model.
    :param no_parameters_objects: Skip wrapping tensors in
     ``torch.nn.Parameters`` objects (~20% speedup, may impact
     some models)
    """
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(name,
                                      torch.nn.Parameter(torch.as_tensor(array)))
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array))


def nbytes(frame, _bytes_like=(bytes, bytearray)):
    if isinstance(frame, _bytes_like):
        return len(frame)
    else:
        try:
            return frame.nbytes
        except AttributeError:
            return len(frame)


def pack_frames_prelude(frames):
    """Pack the `frames` metadata."""
    lengths = [struct.pack("Q", len(frames))] + [
        struct.pack("Q", nbytes(frame)) for frame in frames
    ]
    return b"".join(lengths)


def pack_frames(frames):

    prelude = [pack_frames_prelude(frames)]

    if not isinstance(frames, list):
        frames = list(frames)

    data_ls = prelude + frames
    data_sz = sum(map(lambda b: len(b), data_ls))
    return data_sz, data_ls


def unpack_frames(b):
    """Unpack bytes into a sequence of frames.

    This assumes that length information is at the front of the bytestring,
    as performed by pack_frames

    See Also
    --------
    pack_frames
    """
    (n_frames,) = struct.unpack("Q", b[:8])

    frames = []
    start = 8 + n_frames * 8
    for i in range(n_frames):
        (length,) = struct.unpack("Q", b[(i + 1) * 8: (i + 2) * 8])
        frame = b[start: start + length]
        frames.append(frame)
        start += length

    return frames


def test_pickled_resnet50():
    from psutil import Process
    import os
    import pickle
    import transformers
    from pyarrow import plasma
    from multiprocessing.shared_memory import SharedMemory

    store = plasma.start_plasma_store(1 * 1024 * 1024 * 1024)
    name = store.__enter__()[0]
    client = plasma.connect(name)

    buffer = []
    p = Process(os.getpid())

    transformers.logging.set_verbosity_error()
    model = transformers.BertModel.from_pretrained('bert-base-uncased')

    serialized_model = pickle.dumps(extract_tensors(
        model), buffer_callback=lambda b: buffer.append(b.raw()), protocol=pickle.HIGHEST_PROTOCOL)

    sz, ls = pack_frames(buffer)
    shared_mem = SharedMemory(create=True, size=sz)

    write_offset = 0
    for data in ls:
        write_end = write_offset + len(data)
        shared_mem.buf[write_offset:write_end] = data  # type: ignore

        write_offset = write_end

    serial = client.put(serialized_model)
    del buffer
    del serialized_model

    # Take readings after putting the object in the store
    memory_info_before = p.memory_info().rss

    # Make 100 copies of the object
    # Verified that get(serial) is working fine
    objects = []
    for i in range(100):
        (model, weights) = pickle.loads(
            client.get(serial), buffers=unpack_frames(shared_mem.buf[:sz]))
        objects.append((model, weights))
        replace_tensors(model, weights)
        objects[i][0].eval()

    # Take readings after making 100 copies of the object
    memory_info_after = p.memory_info().rss
    del objects
    gc.collect()

    # Verify that the objects are the same
    # assert memory_info_before == memory_info_after
    print(f"Used Memory {memory_info_before / 1e6} MB")
    print(f"Used Memory {memory_info_after / 1e6} MB")
    shared_mem.close()
    shared_mem.unlink()


if __name__ == "__main__":
    # test_plasma_with_numpy_array()
    # test_pickled_numpy_array()
    test_pickled_resnet50()
