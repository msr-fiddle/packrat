
def get_model(name: str):
    if name == "resnet50":
        import torchvision.models as models
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        return model
    elif name == "inception":
        import torchvision.models as models
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT)
        model.eval()
        return model
    elif name == "bert":
        import transformers
        transformers.logging.set_verbosity_error()
        model = transformers.BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        return model
    elif name == "gpt2":
        import transformers
        model = transformers.GPT2Model.from_pretrained('gpt2')
        model.eval()
        return model
    else:
        raise Exception("Unknown model: {}".format(name))


class Cache:
    def __init__(self):
        self.cache = {}
        self.benchmarks = ["resnet50", "bert", "gpt2", "inception"]
        for benchmark in self.benchmarks:
            self.cache[benchmark] = get_model(benchmark)


if __name__ == '__main__':
    cache = Cache()
    import ray
    ray.put(cache)
    from pyarrow import plasma
    client = plasma.connect("/tmp/plasma")
    client.put(cache)
    print(cache.cache["resnet50"])
    # print(cache.cache["bert"])
    # print(cache.cache["gpt2"])
    # print(cache.cache["inception"])
