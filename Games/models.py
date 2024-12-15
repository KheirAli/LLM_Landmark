import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from itertools import cycle
from pytorch_metric_learning.losses import NTXentLoss
import torch.nn.functional as F
from pytorch_metric_learning.distances import DotProductSimilarity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        
        # Compute pairwise cosine similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create mask for positive samples
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Compute logits
        logits = similarity_matrix / self.temperature
        
        # Mask out self-similarities
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.size(0)).view(-1, 1).to(mask.device),
            0
        )
        logits = logits * logits_mask
        
        # Compute the log softmax
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Only keep positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Compute the loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return loss
    
class ContrastiveDataset(Dataset):
    def __init__(self, data_set, max_shape):
        self.data_set = self.pad_to_max_shape(data_set, max_shape)

    def pad_to_max_shape(self, arrays, max_shape):
        padded_arrays = []
        for array in arrays:
            # print(torch.stack(array).shape, max_shape)
            pad_width = [(0, max_shape[i] - torch.stack(array).shape[i]) for i in range(len(torch.stack(array).shape))]
            pad_width_flat = [item for sublist in reversed(pad_width) for item in sublist]
            padded_array = torch.nn.functional.pad(torch.stack(array), pad_width_flat, 'constant', 0)
            padded_arrays.append(padded_array)
        return padded_arrays

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]
    
class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_1):
        super(ContrastiveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim_1)
        self.fc3 = nn.Linear(hidden_dim_1, 1)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        return x
    
class ContrastiveModel_label(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_1):
        super(ContrastiveModel_label, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim_1)
        self.fc3 = nn.Linear(hidden_dim_1, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.softmax(x, dim = -1)
        return x


with open('2.pkl', 'rb') as file:
    objects = []
    while True:
        try:
            obj = pickle.load(file)
            objects.append(obj)
            break
        except EOFError:
            break


def train_cont_model_label_output(positive_set, negative_set, training_step = 100, loss_func = None, model = None, batch_size = 1, lr = 1e-3, save_name = 'contrastive_model'):
    
    positive_bigger_negative = (len(positive_set)-len(negative_set))>0
    # Initialize the model, optimizer, and loss function
    if len(objects[0]['V'][0]) == 1:
        input_dim = len(objects[0]['V'][0][0])
    else:
        input_dim = len(objects[0]['V'][0])  # assume V is a list of vectors
    print('input_dim: ', input_dim)
    hidden_dim = 128
    hidden_dim_1 = 64
    if model is None:
        model = ContrastiveModel_label(input_dim, hidden_dim, hidden_dim_1)
        model.to(device)
        
    else:
        model = model
        
    model.to(device)
    # print('batchsize:', batch_size)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = loss_func
    print(len(negative_set[0][0].shape))
    negative_new_set = []
    if len(negative_set[0][0].shape)>1:
        for i in negative_set:
            negative_new_set.append([tensor.squeeze(0) for tensor in i])
        negative_set = negative_new_set
    # print(len(negative_set[0][0].shape))
    max_shape = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).squeeze(dim=1).shape for arr in positive_set + negative_set)))
    positive_dataset = ContrastiveDataset(positive_set, max_shape)
    negative_dataset = ContrastiveDataset(negative_set, max_shape)
    


    for epoch in tqdm(range(training_step)):
        positive_dataloader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=True)
        negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=True)

        if positive_bigger_negative:
            negative_dataloader = cycle(negative_dataloader)
        else:
            positive_dataloader = cycle(positive_dataloader)

        for positive_samples, negative_samples in zip(positive_dataloader, negative_dataloader):
            positive_samples, negative_samples = positive_samples.to(device), negative_samples.to(device)
            # print(positive_samples.shape, negative_samples.shape)
            # print('positive_samples:', positive_samples.device)
            positive_samples = positive_samples.squeeze()
            negative_samples = negative_samples.squeeze()
            # print(positive_samples.shape)
            z1 = model(positive_samples)
            z2 = model(negative_samples)
            # print(torch.sum(z1))
            # print(z1.shape)
            
            if type(loss_func)== list:
                # print('loss_func:', loss_func[0](z1), loss_func[1](z2))
                loss = loss_func[0](z1) + loss_func[1](z2)
            else:
                loss = criterion(z1, z2, tau = 1)
            # print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
    torch.save(model, save_name)


def train_cont_model(positive_set, negative_set, training_step = 100, loss_func = None, model = None, batch_size = 1, lr = 1e-3, save_name = 'contrastive_model'):
    
    positive_bigger_negative = (len(positive_set)-len(negative_set))>0
    # Initialize the model, optimizer, and loss function
    if len(positive_set[0][0]) == 1:
        input_dim = len(positive_set[0][0][0])
    else:
        input_dim = len(positive_set[0][0])  # assume V is a list of vectors
    print('input_dim: ', input_dim)
    hidden_dim = 128
    hidden_dim_1 = 64
    if model is None:
        model = ContrastiveModel(input_dim, hidden_dim, hidden_dim_1)
        model.to(device)
        
    else:
        model = model
        
    model.to(device)
    # print('batchsize:', batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = loss_func
    print(len(negative_set[0][0].shape))
    negative_new_set = []
    if len(negative_set[0][0].shape)>1:
        for i in negative_set:
            negative_new_set.append([tensor.squeeze(0) for tensor in i])
        negative_set = negative_new_set
    # print(len(negative_set[0][0].shape))
    max_shape = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).squeeze(dim=1).shape for arr in positive_set + negative_set)))
    positive_dataset = ContrastiveDataset(positive_set, max_shape)
    negative_dataset = ContrastiveDataset(negative_set, max_shape)
    


    for epoch in tqdm(range(training_step)):
        positive_dataloader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=True)
        negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=True)

        if positive_bigger_negative:
            negative_dataloader = cycle(negative_dataloader)
        else:
            positive_dataloader = cycle(positive_dataloader)


        for positive_samples, negative_samples in zip(positive_dataloader, negative_dataloader):
            positive_samples, negative_samples = positive_samples.to(device), negative_samples.to(device)
            # print(positive_samples.shape, negative_samples.shape)
            # print('positive_samples:', positive_samples.device)
            positive_samples = positive_samples.squeeze()
            negative_samples = negative_samples.squeeze()
            # print(positive_samples.shape)
            z1 = model(positive_samples)
            z2 = model(negative_samples)
            # print(torch.sum(z1))
            # print(z1.shape)
            
            if type(loss_func)== list:
                # print('loss_func:', loss_func[0](z1), loss_func[1](z2))
                loss = loss_func[0](z1) + loss_func[1](z2)
            else:
                loss = criterion(z1, z2, tau = epoch)
            # print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
    torch.save(model, save_name)

class ContrastiveModel_feature(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_1):
        super(ContrastiveModel_feature, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim_1)
        self.fc3 = nn.Linear(hidden_dim_1, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


def contrastive_loss_feature(z1, z2, z3, tau=0.5):
    min_size = min(z1.size(0), z2.size(0), z3.size(0))
    z1 = z1[:min_size]/z1.norm(dim=2, keepdim=True).to(device)
    z2 = z2[:min_size]/z2.norm(dim=2, keepdim=True).to(device)
    z3 = z3/z3.norm(dim=1, keepdim=True).to(device)
    # z3 = z3.view(1,1,100)
    print('HERE:',z1.shape, z2.shape, z3.shape)
    # z1_transposed = z1.transpose(1, 2)/z1.norm()
    # norm_z3 = norm_B.transpose(0, 1).unsqueeze(2)
    distances_z1 = torch.norm(z1-z3)
    # z2_transposed = z2.transpose(1, 2)/z2.norm()
    distances_z2 = torch.norm(z2-z3)
    # distances_z1_z2 = torch.norm(z2_transposed, z2_transposed)
    # distances_z2 = torch.cdist(z2_transposed, z2_transposed)
    # distances_z1_z2 = torch.cdist(z1_transposed, z2_transposed)

    score = distances_z1 - distances_z2# - 2*distances_z1_z2.sum()
    # print(torch.max(z1-z2), torch.min(score))
    # print('score:', score)
    return score


def train_cont_model_feature(positive_set, negative_set, ancher, training_step = 100, loss_func = contrastive_loss_feature, model = None, batch_size = 1, lr = 1e-3, save_name = 'contrastive_model'):
    
    positive_bigger_negative = (len(positive_set)-len(negative_set))>0
    # Initialize the model, optimizer, and loss function
    if len(objects[0]['V'][0]) == 1:
        input_dim = len(objects[0]['V'][0][0])
    else:
        input_dim = len(objects[0]['V'][0])  # assume V is a list of vectors
    print('input_dim: ', input_dim)
    hidden_dim = 128
    hidden_dim_1 = 64
    if model is None:
        model = ContrastiveModel_feature(input_dim, hidden_dim, hidden_dim_1)
        model.to(device)
        
    else:
        model = model
        
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = loss_func
    # print(positive_set[0])
    max_shape = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).shape for arr in positive_set + negative_set)))
    max_shape_ancher = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).shape for arr in ancher)))
    print(max_shape)
    # print([torch.stack(arr).shape for arr in positive_set + negative_set])

    positive_dataset = ContrastiveDataset(positive_set, max_shape)
    negative_dataset = ContrastiveDataset(negative_set, max_shape)
    ancher_dataset = ContrastiveDataset(ancher, max_shape_ancher)
    positive_dataloader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=True)
    negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=True)
    ancher_dataloader = DataLoader(ancher_dataset, batch_size=batch_size, shuffle=True)
    if positive_bigger_negative:
        negative_dataloader = cycle(negative_dataloader)
    else:
        positive_dataloader = cycle(positive_dataloader)

    for epoch in tqdm(range(training_step)):

        for positive_samples, negative_samples, ancher_sample in zip(positive_dataloader, negative_dataloader, ancher_dataloader):
            positive_samples, negative_samples, ancher_sample = positive_samples.to(device), negative_samples.to(device), ancher_sample.to(device)
            # print(positive_samples.shape)
            # print('positive_samples:', positive_samples.device)
            positive_samples = positive_samples.squeeze()
            negative_samples = negative_samples.squeeze()
            ancher_sample = ancher_sample.squeeze()
            # print(positive_samples.shape)
            z1 = model(positive_samples)
            z2 = model(negative_samples)
            z3 = model(ancher_sample)
            # print(torch.sum(z1))
            
            if type(loss_func)== list:
                # print('loss_func:', loss_func[0](z1), loss_func[1](z2))
                loss = loss_func[0](z1) + loss_func[1](z2)
            else:
                loss = criterion(z1, z2, z3, tau = 1)
            # print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
    torch.save(model, save_name)


def train_cont_model_all_loss(positive_set, negative_set, positive_state_set = None, negative_state_set = None, training_step = 100, loss_func = None, model = None, batch_size = [1,1], lr = 1e-3, save_name = 'contrastive_model'):
    batch_size_1 = batch_size[0]
    batch_size_2 = batch_size[1]
    positive_bigger_negative = (len(positive_set)-len(negative_set))>0
    positive_bigger_negative_state = (len(positive_state_set)-len(negative_state_set))>0
    # Initialize the model, optimizer, and loss function
    if len(objects[0]['V'][0]) == 1:
        input_dim = len(objects[0]['V'][0][0])
    else:
        input_dim = len(objects[0]['V'][0])  # assume V is a list of vectors
    print('input_dim: ', input_dim)
    hidden_dim = 128
    hidden_dim_1 = 64
    if model is None:
        model = ContrastiveModel(input_dim, hidden_dim, hidden_dim_1)
        model.to(device)
        
    else:
        model = model
        
    model.to(device)
    # print('batchsize:', batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = loss_func
    # print(len(negative_set[0][0].shape))
    negative_new_set = []
    if len(negative_set[0][0].shape)>1:
        for i in negative_set:
            negative_new_set.append([tensor.squeeze(0) for tensor in i])
        negative_set = negative_new_set
    # print(len(negative_set[0][0].shape))
    max_shape = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).squeeze(dim=1).shape for arr in positive_set + negative_set)))
    positive_dataset = ContrastiveDataset(positive_set, max_shape)
    negative_dataset = ContrastiveDataset(negative_set, max_shape)
    
    positive_dataloader = DataLoader(positive_dataset, batch_size=batch_size_1, shuffle=True)
    negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size_1, shuffle=True)

    positive_dataset_state = ContrastiveDataset(positive_state_set, max_shape)
    negative_dataset_state = ContrastiveDataset(negative_state_set, max_shape)
    
    positive_dataloader_state = DataLoader(positive_dataset_state, batch_size=batch_size_2, shuffle=True)
    negative_dataloader_state = DataLoader(negative_dataset_state, batch_size=batch_size_2, shuffle=True)
    if positive_bigger_negative:
        negative_dataloader = cycle(negative_dataloader)
    else:
        positive_dataloader = cycle(positive_dataloader)
    
    if positive_bigger_negative_state:
        negative_dataloader_state = cycle(negative_dataloader_state)
    else:
        positive_dataloader_state = cycle(positive_dataloader_state)

    for epoch in tqdm(range(training_step)):

        for positive_samples, negative_samples, positive_samples_state, negative_samples_state in zip(positive_dataloader, negative_dataloader,positive_dataloader_state, negative_dataloader_state):
            positive_samples, negative_samples, positive_samples_state, negative_samples_state = positive_samples.to(device), negative_samples.to(device), positive_samples_state.to(device), negative_samples_state.to(device)
            # print(positive_samples.shape, negative_samples.shape)
            # print('positive_samples:', positive_samples.device)
            positive_samples = positive_samples.squeeze()
            negative_samples = negative_samples.squeeze()
            positive_samples_state = positive_samples_state.squeeze()
            negative_samples_state = negative_samples_state.squeeze()
            # print(positive_samples.shape)
            z1 = model(positive_samples)
            z2 = model(negative_samples)
            z3 = model(positive_samples_state)
            z4 = model(negative_samples_state)
            # print(torch.sum(z1))
            muo = (1/(epoch+10))
            muo = 2
            loss = muo*loss_func[0](z3) + muo*loss_func[1](z4) + loss_func[2](z1, z2)
            print(loss_func[0](z3),loss_func[1](z4),loss_func[2](z1, z2))
            # print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
    torch.save(model, save_name)

def train_cont_model_cross_entropy(positive_set, negative_set, training_step = 100, model = None, batch_size = 1, lr = 1e-3, save_name = 'contrastive_model'):
    
    positive_bigger_negative = (len(positive_set)-len(negative_set))>0
    # Initialize the model, optimizer, and loss function
    if len(objects[0]['V'][0]) == 1:
        input_dim = len(objects[0]['V'][0][0])
    else:
        input_dim = len(objects[0]['V'][0])  # assume V is a list of vectors
    print('input_dim: ', input_dim)
    hidden_dim = 128
    hidden_dim_1 = 64
    if model is None:
        model = ContrastiveModel(input_dim, hidden_dim, hidden_dim_1)
        model.to(device)
        
    else:
        model = model
        
    model.to(device)
    # print('batchsize:', batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(len(negative_set[0][0].shape))
    negative_new_set = []
    if len(negative_set[0][0].shape)>1:
        for i in negative_set:
            negative_new_set.append([tensor.squeeze(0) for tensor in i])
        negative_set = negative_new_set
    # print(len(negative_set[0][0].shape))
    max_shape = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).squeeze(dim=1).shape for arr in positive_set + negative_set)))
    positive_dataset = ContrastiveDataset(positive_set, max_shape)
    negative_dataset = ContrastiveDataset(negative_set, max_shape)
    


    for epoch in tqdm(range(training_step)):
        positive_dataloader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=True)
        negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=True)

        if positive_bigger_negative:
            negative_dataloader = cycle(negative_dataloader)
        else:
            positive_dataloader = cycle(positive_dataloader)
        for positive_samples, negative_samples in zip(positive_dataloader, negative_dataloader):
            positive_samples, negative_samples = positive_samples.to(device), negative_samples.to(device)
            # print(positive_samples.shape, negative_samples.shape)
            # print('positive_samples:', positive_samples.device)
            positive_samples = positive_samples.squeeze()
            negative_samples = negative_samples.squeeze()
            # print(positive_samples.shape)
            z1 = model(positive_samples)
            z2 = model(negative_samples)
            # print(z1.shape())
            logits_pos = z1.sum(dim=1)
            logits_neg = z2.sum(dim=1)
            # print((logits_pos).sum(), logits_neg.sum())
            pos_target = torch.ones([logits_pos.size()[0],]).to(device)
            # print(f'pos len sis: {pos_target.size()}')
            neg_target = torch.zeros([logits_neg.size()[0],]).to(device)
            
            logits = torch.cat((logits_pos, logits_neg), dim=0)
            targets = torch.cat((pos_target, neg_target), dim=0)
            # print((logits).size(), targets.size())
            # loss_func = nn.CrossEntropyLoss()
            loss_func = NTXentLoss(temperature=0.10)#, distance=DotProductSimilarity())
            loss = loss_func(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
    torch.save(model, save_name)
