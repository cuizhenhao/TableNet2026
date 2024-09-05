class Net_Info:
    increment_id = 0
    save_path = None

    @staticmethod
    def newId():
        Net_Info.increment_id += 1
        return Net_Info.increment_id

    @staticmethod
    def get_save_path():
        return Net_Info.save_path

    @staticmethod
    def set_save_path(save_path):
        Net_Info.save_path = save_path
        return Net_Info.save_path

    def __init__(self, layer_name, previous):
        self.first_codebook = None
        self.id = Net_Info.newId()
        self.net_type = None
        self.save_path = Net_Info.save_path
        self.layer_name = layer_name
        self.previous = previous

    def __str__(self):  # 定义打印对象时打印的字符串
        return self.get_name()

    def get_name(self):
        return str(self.id) + '-' + str(self.layer_name)

    def get_previous_codebook(self):
        if self.previous is None:  # 开始节点
            result = self.first_codebook
        elif isinstance(self.previous, tuple):
            l = [i.last_codebook for i in self.previous]
            result = tuple(l)
        else:
            result = self.previous.last_codebook
        return result

    def set_first_codebook(self, first_codebook):
        self.first_codebook = first_codebook
