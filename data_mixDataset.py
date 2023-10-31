import torch


class mixDataset(torch.utils.data.Dataset):
    """
    Class to use both UVDoc and Doc3D datasets at the same time.
    """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, ii):
        if len(self.datasets[0]) < len(self.datasets[1]):
            len_shortest = len(self.datasets[0])
            i_shortest = ii % len_shortest
            return self.datasets[0][i_shortest], self.datasets[1][ii]
        else:
            len_shortest = len(self.datasets[1])
            jj = ii % len_shortest
            return self.datasets[0][ii], self.datasets[1][jj]

    def __len__(self):
        return max(len(d) for d in self.datasets)
