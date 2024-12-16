
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from itertools import cycle, islice
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
class ContrastiveDataset_without_repeat(Dataset):
    def __init__(self, data_set, max_shape):
        self.data_set = self.pad_to_max_shape(data_set, max_shape)

    def pad_to_max_shape(self, arrays, max_shape):
        padded_arrays = []
        for array in arrays:
            # print(torch.stack(array).shape, max_shape)
            stacked_array = torch.unique(torch.stack(array), dim=0)
            # stacked_array = torch.stack(array)
            pad_width = [(0, max_shape[i] - stacked_array.shape[i]) for i in range(len(stacked_array.shape))]
            pad_width_flat = [item for sublist in reversed(pad_width) for item in sublist]
            
            # print('padded_array:', stacked_array.shape)
            padded_array = torch.nn.functional.pad(stacked_array, pad_width_flat, 'constant', 0)
            # print('padded_array:', padded_array.shape)
            # padded_array = torch.unique(padded_array, dim=0)
            # print('padded_array:', padded_array.shape)
            # padded_arrays.append(stacked_array)
            padded_arrays.append(padded_array)
        return padded_arrays

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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


class ContrastiveDataset(Dataset):
    def __init__(self, data_set, max_shape):
        self.data_set = self.pad_to_max_shape(data_set, max_shape)

    def pad_to_max_shape(self, arrays, max_shape):
        padded_arrays = []
        for array in arrays:
            # print(torch.stack(array).shape, max_shape)
            # stacked_array = torch.unique(torch.stack(array), dim=0)
            stacked_array = torch.stack(array)
            pad_width = [(0, max_shape[i] - stacked_array.shape[i]) for i in range(len(stacked_array.shape))]
            pad_width_flat = [item for sublist in reversed(pad_width) for item in sublist]
            
            # print('padded_array:', stacked_array.shape)
            padded_array = torch.nn.functional.pad(stacked_array, pad_width_flat, 'constant', 0)
            # print('padded_array:', padded_array.shape)
            # padded_array = torch.unique(padded_array, dim=0)
            # print('padded_array:', padded_array.shape)
            padded_arrays.append(padded_array)
            # padded_arrays.append(padded_array)
        return padded_arrays

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]

def my_collate_fn(batch):
    # Instead of stacking, just return the list of tensors
    return batch

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


    max_shape = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).squeeze(dim=1).shape for arr in positive_set + negative_set)))
    with torch.no_grad():
        positive_dataset = ContrastiveDataset_without_repeat(positive_set, max_shape)
        # print('FUCK')
        negative_dataset = ContrastiveDataset_without_repeat(negative_set, max_shape)
    


    for epoch in tqdm(range(training_step)):
        positive_dataloader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=True)#, collate_fn=my_collate_fn)
        negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=True)#, collate_fn=my_collate_fn)

        if positive_bigger_negative:
            negative_dataloader = cycle(negative_dataloader)
        else:
            positive_dataloader = cycle(positive_dataloader)

        max_iterations = max(len(positive_dataset), len(negative_dataset))
        for positive_samples, negative_samples in zip(positive_dataloader, negative_dataloader):
            # positive_samples, negative_samples = torch.stack(positive_samples).to(device), torch.stack(negative_samples).to(device)
            positive_samples, negative_samples = positive_samples.to(device), negative_samples.to(device)
            # print(positive_samples.shape, negative_samples.shape)
            # print('positive_samples:', positive_samples.device)
            

            positive_samples = positive_samples.squeeze()
            negative_samples = negative_samples.squeeze()

            # zero_slices = torch.all(positive_samples == 0, dim=2)
            # # print('zero_slices:', zero_slices.shape)
            # positive_samples = [a[~z] for a, z in zip(positive_samples, zero_slices)]
            # zero_slices = torch.all(negative_samples == 0, dim=2)
            # negative_samples = negative_samples[~zero_slices,:]
            if batch_size == 1:
                positive_samples = positive_samples.unsqueeze(0)
                negative_samples = negative_samples.unsqueeze(0)
            
            
            
            # mask = torch.zeros_like(positive_samples)
            # mask[:,:,119:123] = 1 # for pickup
            # # mask[:,:,123:179] = 1
            # # mask[:,:,2:58] = 1      # for closeby
            # mask[:,:,182:] = 1      #type
            # positive_samples = positive_samples * mask


            # mask = torch.zeros_like (negative_samples)
            # mask[:,:,119:123] = 1
            # # mask[:,:,123:179] = 1 # for reach
            # # mask[:,:,2:58] = 1
            # mask[:,:,182:] = 1
            # negative_samples = negative_samples * mask
            
            mask = torch.abs(positive_samples).sum(dim=2) > 0
            mask = mask.unsqueeze(dim=-1) * 1

            # positive_samples = positive_samples + 1e-4 * torch.randn(positive_samples.size(), requires_grad=False)
            
            z1 = model(positive_samples)
            # mask = F.pad(mask, (0, 0, 0, z1.size(1) - mask.size(1)))
            # z1 = z1[mask]
            
            z1 = z1* mask
            # exit()
            
            mask = torch.abs(negative_samples).sum(dim=2) > 0
            mask = mask.unsqueeze(dim=-1) * 1

            # negative_samples = negative_samples + 1e-4 * torch.randn(negative_samples.size(), requires_grad=False)

            z2 = model(negative_samples)
            # mask = F.pad(mask, (0, 0, 0, z2.size(1) - mask.size(1)))
            # print('Z1:', z1.shape, z2.shape, mask.shape)
            
            z2 = z2* mask
            # print(z1.shape, model(torch.zeros_like(positive_samples)).shape, torch.zeros_like(positive_samples[:,:,0]).shape)
            
            if type(loss_func)== list:
                # print('loss_func:', loss_func[0](z1), loss_func[1](z2))
                loss = loss_func[0](z1) + loss_func[1](z2)
            else:
                loss = criterion(z1, z2, tau = epoch) #+ torch.nn.functional.mse_loss(model(torch.zeros_like(positive_samples)),torch.zeros_like(positive_samples[:,:,0]).unsqueeze(dim=-1))
            # print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{training_step}], Loss: {loss.item()}')
    torch.save(model, save_name)

def pre_process_data(file_name, print_yes = False, save_yes = False):
    positive_set = []
    negative_set = []
    negative_set_enemy = []
    positive_set_state = []
    negative_set_state = []
    negative_set_enemy_state = []
    with open(file_name, 'rb') as file:
        objects = []
        while True:
            try:
                obj = pickle.load(file)
                label = obj['label']
                objects.append(obj)
                V = obj['V']
                if label == 1:
                    negative_set.append(V)
                    negative_set_state.append(obj['state_expl_agent'])
                elif label == 2:
                    positive_set.append(V)
                    positive_set_state.append(obj['state_expl_agent'])
                else:
                    negative_set_enemy.append(V)
                    negative_set_enemy_state.append(obj['state_expl_agent'])
            except EOFError:
                break
    if print_yes:
        print(f'We have {len(objects)} data at {file_name}')
        print(f'Positive set: {len(positive_set)}')
        print(f'Negative set: {len(negative_set)}')
        print(f'Negative set enemy: {len(negative_set_enemy)}')

    if save_yes:
        torch.save(positive_set, 'positive_set_test.pth')
        torch.save(negative_set, 'negative_set_test.pth')
        torch.save(negative_set_enemy, 'negative_set_enemy_test.pth')   
        torch.save(positive_set_state, 'positive_set_state_test.pth')
        torch.save(negative_set_state, 'negative_set_state_test.pth')
        torch.save(negative_set_enemy_state, 'negative_set_enemy_state_test.pth')   

    return objects, positive_set, negative_set, negative_set_enemy, positive_set_state, negative_set_state, negative_set_enemy_state

def dimension_check(positive_set, negative_set, negative_set_enemy, positive_set_pretrain, negative_set_pretrain):
    positive_set_new = []
    negative_set_new = []
    negative_set_enemy_new = []
    positive_set_pretrain_new = []
    negative_set_pretrain_new = []

    for i in positive_set:
        positive_set_new.append([tensor.squeeze(0) for tensor in i])
    for i in negative_set:
        negative_set_new.append([tensor.squeeze(0) for tensor in i])
    for i in negative_set_enemy:
        negative_set_enemy_new.append([tensor.squeeze(0) for tensor in i])
    positive_set = positive_set_new
    negative_set = negative_set_new
    negative_set_enemy = negative_set_enemy_new

    for i in positive_set_pretrain:
        positive_set_pretrain_new.append([tensor.squeeze(0) for tensor in i])
    for i in negative_set_pretrain:
        negative_set_pretrain_new.append([tensor.squeeze(0) for tensor in i])
    positive_set_pretrain = positive_set_pretrain_new
    negative_set_pretrain = negative_set_pretrain_new

    return positive_set, negative_set, negative_set_enemy, positive_set_pretrain, negative_set_pretrain

def postitive_check(positive_state, postitive_set, key_word='pickup_key'):
    """
    Checks and removes positive samples that do not contain a specific keyword.

    Args:
        positive_state (list): The state of positive samples.
        postitive_set (list): The positive samples.
        key_word (str, optional): The keyword to check for. Defaults to 'pickup_key'.

    Returns:
        tuple: A tuple containing the updated positive state and positive set.
    """
    index = []
    for i in range(len(positive_state)):
        counter = 0
        for j in range(len(positive_state[i])):
            if key_word in positive_state[i][j][0]:
                counter += 1
        if counter > 0:
            index.append(i)
            # print('Index:', i)

    for idx in sorted(index, reverse=True):
        del positive_state[idx]
    for idx in sorted(index, reverse=True):
        del postitive_set[idx]

    return positive_state, postitive_set

def negative_check(negative_state, negative_set, key_word='pickup_key'):
    index = []
    for i in range(len(negative_state)):
        counter = 0
        for j in range(len(negative_state[i])):
            if key_word in negative_state[i][j][0]:
                counter += 1
        if counter > 0:
            index.append(i)
            # print('Index:', i)

    for idx in sorted(index, reverse=True):
        del negative_state[idx]
    for idx in sorted(index, reverse=True):
        del negative_set[idx]

    return negative_state, negative_set


def pretrain_pre_process(positive_state, postitive_set, key_word='pickup_key'):
    positive_state_new = []
    positive_set_new = []
    negative_state_new = []
    negative_set_new = []
    for i in range(len(positive_state)):
        for j in range(len(positive_state[i])):
            if key_word in positive_state[i][j][0]:
                positive_state_new.append(positive_state[i][j])
                positive_set_new.append(postitive_set[i][j])
            else:
                negative_state_new.append(positive_state[i][j])
                negative_set_new.append(postitive_set[i][j])

    return positive_state_new, positive_set_new, negative_state_new, negative_set_new



def postitive_check_length(positive_state, key_word='pickup_key'):
    """
    Checks the length of positive samples that contain a specific keyword.

    Args:
        positive_state (list): The state of positive samples.
        key_word (str, optional): The keyword to check for. Defaults to 'pickup_key'.
    """
    index = []
    for i in range(len(positive_state)):
        counter = 0
        for j in range(len(positive_state[i])):
            if key_word in positive_state[i][j][0]:
                counter += 1
        if counter == 0:
            index.append(i)
            # print('Index:', i)

    print('Length:', len(positive_state) - len(index))

    return 
file_saved_before = True
import sys 
import os
current_directory = os.getcwd()

# Print the current directory

file_path = __file__

# Get the directory of the current file
file_directory = os.path.dirname(file_path)

# Optionally, convert it to an absolute path if needed
absolute_directory = os.path.abspath(file_directory)


sys.path.append('..')
# objects = pre_process_data('1.pkl', print_yes = True)
if not(file_saved_before):
    objects_full, positive_set, negative_set, negative_set_enemy, positive_set_state, negative_set_state, negative_set_enemy_state = pre_process_data('2.pkl', print_yes = True, save_yes= True)
else:
    positive_set = torch.load(file_directory+'/P.pth')
    # negative_set = torch.load('negative_set_test.pth')
    # negative_set_enemy = torch.load('negative_set_enemy_test.pth')
    positive_set_state = torch.load(file_directory+'/PS.pth')
    # negative_set_state = torch.load('negative_set_state_test.pth')
    # negative_set_enemy_state = torch.load('negateive_set_enemy_state_test.pth')
    # positive_set = torch.load('positive_small.pth')
    negative_set = torch.load(file_directory+'/N.pth')
    negative_set_enemy = torch.load(file_directory+'/NE.pth')
    # positive_set_state = torch.load('positive_state_small.pth')
    negative_set_state = torch.load(file_directory+'/NS.pth')
    negative_set_enemy_state = torch.load(file_directory+'/NES.pth') # when you are running for the second time, just fix this typo in the name of the file

   



# objects_2 = pre_process_data('14.pkl', print_yes = True)
# positive_set_state, positive_set = postitive_check(positive_set_state, positive_set, key_word = 'pickup_red_key')
# negative_set_state, negative_set = negative_check(negative_set_state, negative_set, key_word = 'pickup_key')
# negative_set_enemy_state, negative_set_enemy = negative_check(negative_set_enemy_state, negative_set_enemy, key_word = 'pickup_key')
print('HERE:',len(positive_set))
postitive_check_length(negative_set_state, key_word = 'pickup_key')


print(f'Length of positive set: {len(positive_set)}')
print(f'Length of negative set: {len(negative_set)}')
print(f'Length of negative set enemy: {len(negative_set_enemy)}')
print(f'Length of positive set state: {len(positive_set_state)}')
print(f'Length of negative set state: {len(negative_set_state)}')
print(f'Length of negative set enemy state: {len(negative_set_enemy_state)}')
# Print all objects retrieved from the file
# positive_state_pretrain, positive_set_pretrain, negative_state_pretrain, negative_set_pretrain = pretrain_pre_process(positive_set_state+negative_set_enemy_state+negative_set_state, positive_set+negative_set_enemy+negative_set, key_word = 'pickup_key')

# print(f'Length of negative set state for pretrain: {len(negative_state_pretrain)}')
# print(f'Length of positive set state for pretrain: {len(positive_state_pretrain)}')
print(f'len of positive set pretrain: {len(positive_set[0][0][0])}')
count = 0
for i in positive_set:
    count += (len(i))
print(f'Count: {count}')

positive_set_pretrain = positive_set
negative_set_pretrain = negative_set
if len(positive_set[0][0]) == 1:
    positive_set, negative_set, negative_set_enemy, positive_set_pretrain, negative_set_pretrain = dimension_check(positive_set, negative_set, negative_set_enemy, positive_set_pretrain, negative_set_pretrain)


# Define the contrastive loss function
def positive_loss(z1):
    one = torch.ones(z1.size()).to(device)
    z1 = z1.to(device)
    # penalty = torch.abs(torch.abs(z1 - 0.5) - 0.5).mean()
    # print('Z!',z1.shape)
    return torch.nn.functional.mse_loss(z1,one) #+ penalty

def negative_loss(z1):
    one = torch.zeros(z1.size()).to(device)
    z1 = z1.to(device)
    # penalty = torch.abs(torch.abs(z1 - 0.5) - 0.5).mean()
    # print('Z!',z1.shape)
    return torch.nn.functional.mse_loss(z1,one) #+ penalty



def contrastive_loss_cross(z1, z2, tau=0.5):
    # print('Z1',z1[-2:], z2[-2:])
    min_size = min(z1.size(0), z2.size(0))
    # tau = 2
    
    z1 = z1[:min_size].to(device)
    z2 = z2[:min_size].to(device)
    # penalty = torch.abs(torch.abs(z1 - 0.5) - 0.5).mean()+ torch.abs(torch.abs(z2 - 0.5) -0.5).mean()
    # print(z1[1,:], z1.shape, z2.shape)
    # exit()
    z1 = z1.sum(dim=1)
    lambda_ = 1/1e4
    # print('Z!',z1.shape, z2.shape)
    z2 = z2.sum(dim=1)
    # max_1 = z1.max()
    # max_2 = z2.max()
    # # print('Z!',z1.shape, z2.shape)
    # # deifference_from_landmarks = torch.norm(torch.matmul(z1, z1.T), p =2)/torch.norm(z1,p=2)**2
    # max_ = 0
    # if max_1 > max_2:
    #     max_ = max_1
    # else:
    #     max_ = max_2
    # if max_ > 10:
    

    # z1 = z1-max_
    # z2 = z2-max_

    criterion = nn.CrossEntropyLoss()
    pos_target = torch.zeros(z1.size(0)).long().to(device)
    # neg_target = torch.zeros(z2.size(0)).long().to(device)
    # target = torch.cat((pos_target, neg_target), 1)
    Z = torch.cat((z1, z2), 1).float()
    # Z1 = torch.cat((z2, z1), 0).float()
    # Z = torch.cat((Z, Z1), 1).float()
    # print('HERE:',Z.shape, pos_target.shape)
    score = criterion(Z, pos_target)#+ torch.sum((torch.abs(z1-z1.mean())>2).float()) #+ (z1 - z1.mean()).mean()/(tau+1)
    # print('Z:', Z.shape, target.shape)
    
    # score = (z1.exp().sum() / (z1.exp().sum() + ((z2).exp().sum()))).log()

    print('HERE:',z1.mean().item(), z2.mean().item(), 'score:', score.item())
    
   
    # print(z1.exp().mean(), z2.exp().mean(), score)
    # score = ((z1).exp() / ((z1).exp() + ((z2).exp()*tau)))
    # print(torch.max(z1-z2), torch.min(score))
    # if tau < 5:
    #     return z1.mean()+ score
    return  score#+z1.mean() #-1*score + torch.abs(max_ - tau) #+ -1* deifference_from_landmarks#+ lambda_*penalty#+ lambda_ * z1.mean() + lambda_*z2.mean()


def load_model(filneame = 'train.pth'):
    model = torch.load(filneame)
    return model
# file_name_here = 'Model_4_landmark_without_enemy.pth'
file_name_here = 'waste_my_life_without_repeat.pth'
# model = load_model(filneame = file_name_here)
# train_cont_model(positive_set_pretrain, negative_set_pretrain, training_step = 20, loss_func = [positive_loss, negative_loss], model = None, batch_size= 256, save_name='pretrain.pth')
# model = load_model(filneame = 'pretrain.pth')
# file_name_here = 'HMMM.pth'


# train_cont_model(positive_set[:50], negative_set_enemy+negative_set, training_step =1000, loss_func = contrastive_loss_cross, model = load_model(filneame = file_name_here), batch_size= 25, lr = 1e-3, save_name=file_name_here)

model = load_model(filneame = file_name_here)

state_dict = model.state_dict()
# for name, param in state_dict.items():
#     print(name)
#     print(param)
#     print(param[0])
#     print(param[1])
#     exit()
choosen_set = positive_set
choosen_set_state = positive_set_state

    # with open("output1.txt", "w") as f:
Indices = list()
for i in range(len(choosen_set)):
    input = torch.stack([tensor.squeeze(0) for tensor in choosen_set[i]]).to(device)
    input = torch.tensor(input, requires_grad=True)
    output = model(input)
    # print(output.shape)
    mask = output>0.1
    indices = torch.nonzero(mask)
    # print(indices)
    # print(output)
    # output = output[mask]
    

    for i in range(len(indices)):
        idx = indices[i][0]
        output[idx].backward(retain_graph=True)
        # print(output[idx].item())
        grad_1 = input.grad
        
        grad_1 = grad_1[idx,:] 
        maximum = torch.max(grad_1)
        if maximum != 0:
            mask = grad_1 > maximum*0.8
            indices1 = torch.nonzero(mask)
            Indices.append(indices1.squeeze().detach().cpu().numpy())
        
        
        # print(grad_1.shape)
        
# print(Indices)

plot_array = np.zeros((1,400))
for i in range(len(Indices)):
    
    if len((Indices[i]).shape) == 0:
        plot_array[0,Indices[i]] += 1
    else:
        for j in (Indices[i]):
            # print(j)
            plot_array[0,j] += 1
        # print(Indices[i][j])
        # print(Indices[i][j].item())
        # print(plot_array)
# plt.stem(np.log(plot_array[0,:]+1))
# plt.xticks([30, 89, 121,150,200], ['closeby','on_left','pickup', 'reach', 'type']) 
# plt.axvline(x=2, color='r', linestyle='--', label='Vertical Line at x=5')
# plt.axvline(x=58, color='r', linestyle='--', label='Vertical Line at x=5')
# plt.axvline(x=119, color='r', linestyle='--', label='Vertical Line at x=5')
# plt.axvline(x=123, color='r', linestyle='--', label='Vertical Line at x=5')
# plt.axvline(x=179, color='r', linestyle='--', label='Vertical Line at x=5')
# plt.xlabel('Data Point Index')
# plt.ylabel('log(1+Count)')
# plt.grid(True)
# plt.show()

l = []
counter = 0
choosen_set = positive_set
choosen_set_state = positive_set_state
l_exp = []

choosen_set = positive_set
choosen_set_state = positive_set_state
threshold = 0.01
threshold = 0.99
all_output = []
with torch.no_grad():
    with open("output_closeby.txt", "w") as f:
        for i in range(len(choosen_set)):
            output = model(torch.stack([tensor.squeeze(0) for tensor in choosen_set[i]]).to(device))
            all_output.append(output)
            l.append((output>threshold).sum().item())
            # print()
            t = ((output>threshold).cpu().squeeze())*1*np.arange(len(output))
            t = t[t>0]
            t = t.tolist()
            # print(t)
            for idx in t:
                l_exp.append(choosen_set_state[i][idx])
            list_of_strings = []
            for idx in (choosen_set_state[i]):
                if len(idx[0]) == 1:
                    list_of_strings.append("\n".join(idx[-1]))
                else:
                    # print('idx:', idx)
                    list_of_strings.append("\n".join(idx))
            # list_of_strings = ["\n".join(arr[0]) for arr in (choosen_set_state[i])]
            for j,line in enumerate(list_of_strings):
                f.write(f'output: {output[j].item()}')
                f.write(line )
                
            f.write("\nnew data \n")

all_pred = [np.arange(2,58), np.arange(119,123), np.arange(123,179), np.arange(182, len(choosen_set[i][0])),np.arange(58,119)]
with torch.no_grad():
    for idx in all_pred:
        all_output_test = []
        temp = torch.ones(len(choosen_set[0][0]))
        temp[idx] = 0
        print('IDX:', idx)
        for i in range(len(choosen_set)):
            test = [(tensor.squeeze(0)) for tensor in choosen_set[i]]
            test = (torch.stack(test).to(device))
            test = test*temp.to(device)
            output = model(test)
            all_output_test.append(output)
        Sum_over_all = 0
        for i in range(len(all_output)):
            # print((all_output[i]-all_output_test[i]).sum().item())
            # break
            Sum_over_all += (torch.abs(all_output[i]-all_output_test[i]).sum())
        print('Sum:', Sum_over_all)      
l_1 = []
choosen_set = negative_set
choosen_set_state = negative_set_state
with torch.no_grad():
    # with open("output1.txt", "w") as f:
    for i in range(len(choosen_set)):
        output = model(torch.stack([tensor.squeeze(0) for tensor in choosen_set[i]]).to(device))
        l_1.append((output>0.99).sum().item())
            # list_of_strings = []
            # for idx in (choosen_set_state[i]):
            #     if len(idx[0]) == 1:
            #         list_of_strings.append("\n".join(idx[-1]))
            #     else:
            #         # print('idx:', idx)
            #         list_of_strings.append("\n".join(idx))
            # # list_of_strings = ["\n".join(arr[0]) for arr in (choosen_set_state[i])]
            # for j,line in enumerate(list_of_strings):
            #     f.write(f'output: {output[j].item()}')
            #     f.write(line )
                
            # f.write("\nnew data \n")
print(l)
# print(l_exp)
# print(len(l),np.ones_like(l))
# plt.scatter(np.ones_like(l), l, color='r')  
# plt.scatter(np.zeros_like(l_1), l_1, color='b')
# plt.show()
with open("output_correct_closeby.txt", "w") as f:  
    for i in l_exp:
        f.write("\n".join(i))
        f.write("\nnew data \n")

exit()   
print(len(choosen_set), len(choosen_set_state[0]))
max_shape = tuple(max(sizes) for sizes in zip(*(torch.stack(arr).squeeze(dim=1).shape for arr in choosen_set)))
positive_dataset = ContrastiveDataset_without_repeat(choosen_set, max_shape)
positive_dataloader = DataLoader(positive_dataset, batch_size=1, shuffle=True)
with torch.no_grad():
    with open("output.txt", "w") as f:
        for i,obj in enumerate(positive_dataloader):
            output = model(obj.squeeze().to(device))
            # output = model(torch.stack([tensor.squeeze(0) for tensor in choosen_set[i]]).to(device))
            # output = model(torch.stack([tensor for tensor in choosen_set[i]]).to(device))
            # print('output:', output)
            list_of_strings = ["\n".join(map(str, arr)) for arr in choosen_set_state[i]]
            # print(list_of_strings)
            # exit()
            for j,line in enumerate(list_of_strings):
                
                f.write(line )
                f.write(f'output: {output[j].item()}')
            f.write("\nnew data \n")
        exit()
        # print('output:', len(output), output)
        # output = output / max_
        # print('output:', len(output), output)
        # output = torch.softmax(output, dim=0)
        
        sorted_output, sorted_indices = torch.sort(output, descending=True)
        # print(sorted_indices)
        index = sorted_indices.squeeze().tolist()
        # print(index)
        if 'pickup_key' in positive_set_state[i][max_index.item()][0]:
            counter += 1
            l.append(1)
        else:
            l.append(0)
      
with open("output.txt", "w") as f:
    for i in range(len(positive_set_state)):
        list_of_strings = ["\n".join(arr) for arr in positive_set_state[i]]
        for line in list_of_strings:
            f.write(line )
        f.write("\nnew data \n")
        for k in range(len(positive_set[i])):
            
        
            
            
            print(output[k], predicted_label[k])
            print(positive_set_state[i][k])
        exit()
import matplotlib.pyplot as plt
print(np.cumsum(np.array(l))[50])
plt.stem('train on 50 data',np.cumsum(np.array(l))[50])
plt.stem('test on 300 data',np.cumsum(np.array(l))[-1] - np.cumsum(np.array(l))[50])
plt.show()






# train_cont_model_feature(positive_set, negative_set, positive_set_pretrain, training_step = 100, loss_func = contrastive_loss_feature, model = None, batch_size= 64, save_name='contrastive_model_feature_for_test_full_1.pth')
# model = load_model(filneame = 'contrastive_model_feature_for_test_full_1.pth')
# model_new = ContrastiveModel((positive_set_pretrain[0][0]).size()[1], 128, 64)

# model_new.fc1.load_state_dict(model.fc1.state_dict())
# model_new.fc2.load_state_dict(model.fc2.state_dict())
# model_new.fc3.load_state_dict(model.fc3.state_dict())

# train_cont_model(positive_set_pretrain, negative_set_pretrain, training_step = 100, loss_func = [positive_loss, negative_loss], model = None, batch_size= 1024, save_name='contrastive_model_for_test_full_1.pth')
# model = load_model(filneame = 'contrastive_model_for_test_full_1.pth')
# with torch.no_grad():
#     output = model(torch.stack([tensor.squeeze(0) for tensor in objects[1]['V']]).to(device))
#     for i in range(len(output)):
#         print('output:', output[i])
#         print(objects[1]['state_expl_agent'][i])
# train_cont_model(positive_set, negative_set_enemy, positive_set_pretrain, negative_set_pretrain, training_step = 200, loss_func = contrastive_loss, model = None, batch_size= 32, lr = 1e-3, save_name='contrastive_model_for_test_1_full_1.pth')
