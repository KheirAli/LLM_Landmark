import torch


class MLP(torch.nn.Module):

    def __init__(self, has_softmax=False, out_size=3, as_dict=False, logic=False, device=None):
        super().__init__()
        self.logic = logic
        self.as_dict = as_dict
        self.device = device
        encoding_base_features = 6
        encoding_entity_features = 9
        encoding_max_entities = 7  #6
        self.num_in_features = encoding_base_features + encoding_entity_features * encoding_max_entities  # 60
        self.num_in_features = 87
        # self.num_in_features = 69

        # self.num_in_features = 87
        modules = [
            torch.nn.Linear(self.num_in_features, 60),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(60, 20),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(20, out_size)
        ]

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, state):
        features = state
        y = self.mlp(features)
        return y
