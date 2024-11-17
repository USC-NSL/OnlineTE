import os
import sys
import zipfile
from urllib.request import urlretrieve
from topologies import (
    TOPOLOGY_ZOO_DIR_NAME, TOPOLOGY_ZOO_ARCHIVE_URL, 
    TOPOLOGIES_PATH
)


def get_topology_zoo(del_zip=False):
    path = os.path.join(TOPOLOGIES_PATH, TOPOLOGY_ZOO_DIR_NAME)
    if os.path.exists(path):
        print("Archive already exists.")
        sys.exit(0)
    
    os.mkdir(path)
    result_path = os.path.join(path, "zoo.zip")
    urlretrieve(TOPOLOGY_ZOO_ARCHIVE_URL, result_path)

    with zipfile.ZipFile(result_path) as zip_file:
        zip_file.extractall(path)
    
    if del_zip:
        os.remove(result_path)


if __name__ == '__main__':
    get_topology_zoo()
