"""
This utility is used to allocate the number of cores on the node.
"""

import platform
import re
import subprocess
import sys


class CPUInfo():
    """
    Parse the machine topology using lscpu command.
    """

    def __init__(self):
        self.cpuinfo = []
        self.mapping = {}
        if platform.system() != "Linux":
            raise RuntimeError("This platform is not supported!!!")

        args = ["lscpu", "--parse=CPU,Core,Node"]
        lscpu_info = subprocess.check_output(
            args, universal_newlines=True).split("\n")

        # Get information about  cpu, core, socket and node
        for line in lscpu_info:
            pattern = r"^([\d]+,[\d]+,[\d]+)"
            regex_out = re.search(pattern, line)
            if regex_out:
                self.cpuinfo.append(regex_out.group(1).strip().split(","))

        for (cpu, core, node) in self.cpuinfo:
            if node not in self.mapping:
                self.mapping[node] = {}
            if core not in self.mapping[node]:
                self.mapping[node][core] = {}
            self.mapping[node][core][cpu] = cpu

    def get_cores_on_node(self, node_id):
        """
        Get the number of cores
        """
        cores = []
        for key in self.mapping[node_id].keys():
            cores.append(int(key))
        cores.sort()
        return cores

    def get_threads_on_node(self, node_id):
        """
        Get the number of cores
        """
        threads = []
        for key in self.mapping[node_id].keys():
            for thread in self.mapping[node_id][key].keys():
                threads.append(int(thread))
        threads.sort()
        return threads

    def get_sockets(self) -> set:
        """
        Get the number of sockets
        """
        return {int(node) for cpu, core, node in self.cpuinfo}


def allocate_cores(mapping: str, threads: int, hyper_threading: bool = False):
    """
    Allocate the number of cores on the node
    """

    info = CPUInfo()
    sockets = info.get_sockets()
    cores = []

    for socket in sockets:
        hts = []
        if hyper_threading:
            hts = info.get_threads_on_node(str(socket))
        else:
            hts = info.get_cores_on_node(str(socket))

        for thread in hts:
            cores.append(thread)

    if mapping == "machine":
        cores.sort()

    return cores[0:threads]


def main():
    """
    Main function
    """
    if len(sys.argv) > 3:
        print("Error: Too many arguments")
        print("Usage: {0} [mapping #cores]".format(sys.argv[0]))
        sys.exit(1)

    if len(sys.argv) == 3:
        mapping = sys.argv[1]
        cores = int(sys.argv[2])
        return allocate_cores(mapping, cores)

    print("Error: Unrecognized affinity mapping '{0}'".format(mapping))
    return []


if __name__ == '__main__':
    print(main())
