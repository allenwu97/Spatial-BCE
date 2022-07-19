import matplotlib.pyplot as plt
from collections import OrderedDict


class PlotLogger(object):
    def __init__(self, params=['loss']):
        self.logger = OrderedDict({param:[] for param in params})
    def update(self, ordered_dict):
        # self.logger.keys()
        assert set(ordered_dict.keys()).issubset(set(self.logger.keys()))
        for key, value in ordered_dict.items():
            self.logger[key].append(value)

    def save(self, file, **kwargs):
        fig, axes = plt.subplots(nrows=len(self.logger), ncols=1)
        for ax, (key, value) in zip(axes, self.logger.items()):
            ax.plot(value)
            ax.set_title(key)

        plt.savefig(file, **kwargs)
        plt.close()
