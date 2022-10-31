

class TestUnitBase(object):
    def __init__(self, name, caches):
        print("=> Performing {}".format(name))
        self.name = name
        self._caches = caches
        self._kv_dict_all = caches.get('kv_dict_all', None)
        self._inf_outputs_data = None

    def gen_kv_dict_all(self, **kwargs):
        pass

    def gen_inf_outputs_data(self, **kwargs):
        pass

    def compute_metric(self, **kwargs):
        pass
