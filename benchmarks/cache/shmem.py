import pickle
import pyarrow
from pyarrow import plasma
from store import Cache
from serde import extract_tensors, replace_tensors


class PlasmaStore:
    def __init__(self) -> None:
        cache = Cache()
        self.store = plasma.start_plasma_store(1024 * 1024 * 1024)
        self.client = plasma.connect(self.store.__enter__()[0])
        resnet = extract_tensors(cache.cache["resnet50"])

        # TODO: Figure out how to serialize the model and tensors
        serialized_model = pickle.dumps(resnet, pickle.HIGHEST_PROTOCOL)
        self.resnetID = self.client.put(serialized_model)
        print("resnetID", self.resnetID)

        # TODO: Figure out how to serialize the model and tensors
        (model, weights) = pickle.loads(self.client.get(self.resnetID))
        replace_tensors(model, weights)
        model.eval()

        pyarrow.SerializationContext.register_type()

if __name__ == "__main__":
    store = PlasmaStore()
