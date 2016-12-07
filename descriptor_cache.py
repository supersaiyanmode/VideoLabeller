import os
import json

class Cache(object):
    def __init__(self, path="cache.txt"):
        self.path = path
        if not os.path.isfile(self.path):
            self.obj = {}
            self.write()
        else:
            with open(self.path) as f:
                self.obj = json.load(f)

    def __contains__(self, key):
        return key in self.obj

    def __getitem__(self, key):
        return self.obj[key]
    
    def __setitem__(self, key, value):
        self.obj[key] = value

    def write(self):
        with open(self.path, "w") as f:
            json.dump(self.obj, f, indent=2)

