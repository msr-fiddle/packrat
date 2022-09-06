from multiprocessing import shared_memory


class Shmem:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        try:
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(
                name=name)

    def __del__(self):
        self.shm.close()
        self.shm.unlink()

    def read(self):
        return self.shm.buf

    def write(self, data):
        self.shm.buf[:] = data


if __name__ == "__main__":
    print(Shmem("test", 1000).name, Shmem("test", 1000).name)
