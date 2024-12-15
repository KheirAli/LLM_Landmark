import torch
from nsfr.utils.common import bool_to_probs
def one_to_prob(z):
    if z > 0.8:
        z = 1
    return 0.99*z

def type(z, a):
    # print('z:', z, 
        #   'a:', a)
    z_type = z[:, 0:7]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
    prob = (a * z_type).sum(dim=1)
    return one_to_prob(prob)


def closeby(z_1, z_2):
    z_1.to(torch.device('cuda:0'))
    z_2.to(torch.device('cuda:0'))
    # print('z_1:', z_1)
    # print('z_2:', z_2)
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]
    existance_1 = z_1[:, 0:7].sum(dim=1)
    existance_2 = z_2[:, 0:7].sum(dim=1) 
    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    dis_y = abs(c_1[:, 1] - c_2[:, 1])

    # result = bool_to_probs(existance_1*existance_2*(dis_x < 2.5))# & (dis_y <= 0.1))
    result = one_to_prob(existance_1*existance_2*(dis_x < 2.5))
    return result

def reach(z_1, z_2):
    z_1.to(torch.device('cuda:0'))
    z_2.to(torch.device('cuda:0'))
    # print('z_1:', z_1)
    # print('z_2:', z_2)
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]
    existance_1 = z_1[:, 0:7].sum(dim=1)
    existance_2 = z_2[:, 0:7].sum(dim=1) 
    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    dis_y = abs(c_1[:, 1] - c_2[:, 1])

    result = one_to_prob(existance_1*existance_2*((dis_x+dis_y) < 0.1))# & (dis_y <= 0.1))

    return result



def on_left(z_1, z_2):
    z_1.to(torch.device('cuda:0'))
    z_2.to(torch.device('cuda:0'))
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    existance_1 = z_1[:, 0:7].sum(dim=1)
    existance_2 = z_2[:, 0:7].sum(dim=1) 
    diff = c_2 - c_1
    result = one_to_prob(existance_1*existance_2*(diff > 0))
    return result


def on_right(z_1, z_2):
    z_1.to(torch.device('cuda:0'))
    z_2.to(torch.device('cuda:0'))
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    diff = c_2 - c_1
    result = bool_to_probs(diff < 0)
    return result


def have_key(z):
    z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    has_key = torch.ones(z.size(dim=0)).to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 1], dim=1).to(torch.device('cuda:0'))
    result = has_key[:] - c[:]
    # print('result:', result)
    return one_to_prob(result)

def open_door(z):
    # z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    z = z[-1]
    has_key = torch.ones(z.size(dim=0)).to(torch.device('cuda:0'))

    c = torch.sum(z[:, :, 2], dim=1).to(torch.device('cuda:0'))
    result = has_key[:] - c[:]

    return one_to_prob(result)


def have_3_coin(z):
    z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    has_key = 3 * torch.ones(z.size(dim=0)).to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 4], dim=1).to(torch.device('cuda:0'))
    result = has_key[:] - c[:]
    result_1 = has_key[:]
    result_1 = result>2
    return bool_to_probs(result_1)

def have_2_coin(z):
    z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    has_key = 2 * torch.ones(z.size(dim=0)).to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 4], dim=1).to(torch.device('cuda:0'))
    result = has_key[:] - c[:]
    result_1 = result>1
    return bool_to_probs(result_1)

def have_1_coin(z):
    z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    has_key = 2 * torch.ones(z.size(dim=0)).to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 4], dim=1).to(torch.device('cuda:0'))
    result = has_key[:] - c[:]
    result_1 = one_to_prob(result>0)
    return result_1 

def have_flag(z):
    z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    has_key = 1 * torch.ones(z.size(dim=0)).to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 5], dim=1).to(torch.device('cuda:0'))
    result = has_key[:] - c[:]

    return one_to_prob(result)

def not_have_key(z):
    z.to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 1], dim=1)
    result = c[:]
    return one_to_prob(result)

def not_have_flag(z):
    z.to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 5], dim=1)
    result = c[:]

    return one_to_prob(result)

def not_have_all_coin(z):
    z.to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 4], dim=1)
    result = c[:]

    return one_to_prob(result)
def pickup_flag(z):
    
    if len(z) == 0:
        return 0
    elif len(z) == 1:
        return have_flag(z[-1])
    else:
        t_1 = have_flag(z[-1])
        t_2 = have_flag(z[-2])
        if t_1 - t_2 >0:
            return 0.99
        else:
            return 0
        # print('pickup:', t_1 - t_2)
        return t_1 - t_2
def pickup_coin(z):
    # z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    # print('z for pickup_key:', len(z))
    if len(z) == 0:
        return 0
    elif len(z) == 1:
        return have_1_coin(z[-1])
    else:
        t_1 = have_1_coin(z[-1])
        t_2 = have_1_coin(z[-2])
        t_3 = have_2_coin(z[-1])
        t_4 = have_2_coin(z[-2])
        if t_1 - t_2 >0 or t_3 - t_4 >0:
            return 0.99
        else:
            return 0
        # print('pickup:', t_1 - t_2)
        return t_1 - t_2
    
def pickup_key(z):
    # z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    # print('z for pickup_key:', len(z))
    if len(z) == 0:
        return 0
    elif len(z) == 1:
        return have_key(z[-1])
    else:
        t_1 = have_key(z[-1])
        t_2 = have_key(z[-2])
        # print('pickup:', t_1 - t_2)
        return t_1 - t_2

def pickup_red_key(z):
    if len(z) == 0:
        return 0
    elif len(z) == 1:
        return have_red_key(z[-1])
    else:
        t_1 = have_red_key(z[-1])
        t_2 = have_red_key(z[-2])
        # print('pickup:', t_1 - t_2)
        return t_1 - t_2
    
def have_red_key(z):
    z.to(torch.device('cuda:0'))
    # print('z for has_key:', z)
    has_key = torch.ones(z.size(dim=0)).to(torch.device('cuda:0'))
    c = torch.sum(z[:, :, 6], dim=1).to(torch.device('cuda:0'))
    result = has_key[:] - c[:]

    return one_to_prob(result)

def safe(z_1, z_2):
    z_1.to(torch.device('cuda:0'))
    z_2.to(torch.device('cuda:0'))
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0])
    result = bool_to_probs(dis_x > 2)
    return result
