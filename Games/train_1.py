
import csv
import os
import sys
import time
from pathlib import Path
from typing import Callable
import math
from inspect import signature
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
from tqdm import tqdm
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from torch.optim import Optimizer, Adam
# print(os.path.relpath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import matplotlib.pyplot as plt

from nudge.agents.logic_agent import LogicPPO
from nudge.agents.neural_agent import NeuralPPO
from nudge.utils import make_deterministic, save_hyperparams
from nudge.env import NudgeBaseEnv

from env_src.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from env_src.getout.getout.getout import Getout
from env_src.getout.getout.actions import GetoutActions
import pickle


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")


OUT_PATH = Path("out/")
IN_PATH = Path("in/")


def create_getout_instance(seed=None):
        # if args.env == 'getoutplus':
        #     enemies = True
        # else:
        #     enemies = False
        # level_generator = DummyGenerator()
        # EVERY TIME RUNNING CHANGE THE PLAYER IN THE GETOUT.py file
        getout = Getout()
        level_generator = ParameterizedLevelGenerator(enemies=False, flag = True, coin = True)
        level_generator.generate(getout, seed=seed)
        getout.render()
        # print(level_generator)
        # print(getout.level.get_representation()['entities'])
        return getout

    # seed = random.randint(0, 100000000)
    # print(seed)

def exp_decay(episode: int):
    """Reaches 2% after about 850 episodes."""
    return max(math.exp(-episode / 500), 0.02)

def env_instance(algorithm: str = "logic",
         environment: str = "getout",
         env_kwargs: dict = None,
         seed: int = 1,
         ):
    make_deterministic(seed)
    if env_kwargs is None:
        env_kwargs = dict()


    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **env_kwargs)
    getout = create_getout_instance()

    return env, getout
def main_agent_creation(algorithm: str = "logic",
         environment: str = "getout",
         env_kwargs: dict = None,
         rules: str = "getout_check",
         seed: int = 10,
        #  device: str = "cuda:0",
         device: str = "cpu",
         epochs: int = 20,
         eps_clip: float = 0.2,
         gamma: float = 0.99,
         optimizer: Optimizer = Adam,
         lr_actor: float = 0.001,
         lr_critic: float = 0.0003,
         epsilon_fn: Callable = exp_decay,
         env = None,
         getout = None,
        ):

    if algorithm == "ppo":
        agent = NeuralPPO(env, lr_actor, lr_critic, optimizer,
                          gamma, epochs, eps_clip, device)
    else:  # logic
        agent = LogicPPO(env, rules, lr_actor, lr_critic, optimizer,
                         gamma, epochs, eps_clip, device)
    n_episodes = 1

    import matplotlib.pyplot as plt
    state_expl = []

    actual_action = []
    
    epsilon = epsilon_fn(n_episodes)

    state = env.extract_logic_state(getout), env.extract_neural_state(getout).to(device)
    action = agent.select_action(state, epsilon=epsilon)
    state_expl.append(agent.policy_old.actor.explaining_staet)
    actual_action.append([action])
    # action_expl.append(agent.policy_old.actor.rule_choosen)

    # state_img = getout.camera.screen
    # reward = getout.step([action])

    # done = getout.level.terminated
    # state, reward, done = env.step(action)
    # done_list.append([done])
    # reward_list.append([reward])
    return state_expl[-1], actual_action[-1], agent.buffer.actions_V[-1], agent.prednames, agent

def agent_action(agent, env, getout, epsilon = 1):
    epsilon = epsilon
    state = env.extract_logic_state(getout), env.extract_neural_state(getout).to(device)
    action = agent.select_action(state, epsilon=epsilon)
    return agent.policy_old.actor.explaining_staet,  [action], agent.buffer.actions_V[-1], agent.prednames

    

def valuefunc(reward_list):
    gamma = 1
    R = 0
    reverse_reward = reward_list[::-1]
    value_list = []
    for i in range(len(reverse_reward)):
        if i == 0:
            value_list.append(reverse_reward[i])
        else:
            value_list.append(reverse_reward[i] + gamma * value_list[i-1])

    return value_list[::-1]

fig, ax = plt.subplots()


information_name = input("Enter saving state information document: ")
# rules_set = ["getout_coin", "getout_bs_rf1"]
rules_set = ["getout_coin"]
# rules_set_tempor = ["getout_check_4", "getout_check_5"]

start_time = time.time()
update_steps = 2
# model = model_instance()
# V_model = V_model_instance()
# for name, param in model.named_parameters():
#     print(f"{name} is on {param.device}")

loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# print(f"Execution time for model: {time.time() - start_time} seconds")
# start_time = time.time()
n_episodes = 100000
agent_list = []
agent_list_tempor = []
stacked_val = []
save_step = 10
env, getout = env_instance()
trajectory_step = 200
for epoch in tqdm(range(n_episodes)):
    tempor_flg = False
    save_state_reward = torch.tensor([]).to(device)
    # prob_out_subtask = torch.tensor([]).to(device)
    reward_list = []
    agent_index = 0
    state_v_list = []
    state_expl_agent = []
    for game_step in range(trajectory_step):
        print("Cursor here:",epoch, game_step)
        actual_action_agent = []
        V_agent = []
        pred_name_agent = []
        
        # print(f"Execution time for 2: {time.time() - start_time} seconds")
        # start_time = time.time()
        if game_step == 0 and epoch == 0:
            for i,rule in enumerate(rules_set):
                
                state_expl, actual_action, V, pred_name, agent = main_agent_creation(rules= rule, env= env, getout= getout)
                agent_list.append(agent)
                if i == 0:
                    state_expl_agent.append(state_expl)
                actual_action_agent.append(actual_action)
                V_agent.append(V[0])
                pred_name_agent.append(pred_name)

                print('HERER: V:', V)
                if i > 0:
                    agent.buffer.clear()
            # for rule in rules_set_tempor:
            #     state_expl, actual_action, V, pred_name, agent = main_agent_creation(rules= rule, env= env, getout= getout)
            #     agent_list_tempor.append(agent)
        else:
            # for agent in agent_list:
            # stateV = agent.policy_old.actor.V_0
            # state_v_list.append(stateV)
            state_expl, actual_action, V, pred_name = agent_action(agent_list[agent_index], env, getout, epsilon=epoch)
            state_expl_agent.append(state_expl)
            actual_action_agent.append(actual_action)
            V_agent.append(V)
            pred_name_agent.append(pred_name)
            # print('V:',  agent.buffer.actions)
        # merged_pred = []
        # merged_val = torch.tensor([]).to(device)


        # _, actual_action, _, _ = agent_action(agent_list[1], env, getout)




        sampled_key = actual_action
        # print(sampled_key, V, pred_name)
        # print(env.extract_logic_state(getout))
        # print('sampled_key:', sampled_key)
        # print(V)
        stateV = agent_list[agent_index].policy_old.actor.V_0
        state_v_list.append(stateV)
        reward = getout.step(sampled_key)
        
        reward_list.append(reward)
        done = getout.level.terminated
        for agent in agent_list:
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
        # print(agent_list[agent_index].buffer.logic_states[-1])
        
        #image part
        state_img = getout.camera.screen
        plt.imshow(state_img)
        plt.pause(0.01)
        plt.show(block=False)
        width, height = state_img.size

# Define the cropping area (left, upper, right, lower)
# We want the bottom half, so the crop starts at height/2
        crop_area = (0, height // 2, width, height)
        state_img = state_img.crop(crop_area)
        # print(state_img.shape)
        img = Image.fromarray(np.array(state_img))
        img.save('image.png')
        exit()



        # state_v = agent_list[agent_index].buffer.state_V[-1]
        
        # if epoch == 1:
        #     print('if this is the same:', state_v == stateV)
        # print('state_v:', state_v)
        
        # print('state_v:', len(state_v_list))
        # print('state_v:', state_v)
        # print('state_exp', state_expl)
        # print(state_expl)
        # if not (game_step == 0 and epoch == 0):
        # if 'not_have_key' in state_expl[0]:
        #     # if game_step == 0 and epoch == 0:
        #     #     break
        #     agent_list[agent_index].buffer.rewards.append(reward)
        #     agent_list[agent_index].buffer.is_terminals.append(done)
        #     # print('not_have_key')
        #     agent_index = 0
        # else:
        #     agent_list[agent_index].buffer.rewards.append(reward)
        #     agent_list[agent_index].buffer.is_terminals.append(done)
        #     agent_index = 1
        label = 0
        
        if done or game_step == trajectory_step - 1:
            
            # print(save_state_reward.shape)
            state_expl, _, _, _ = agent_action(agent_list[agent_index], env, getout, epsilon=epoch)
            state_expl_agent.append(state_expl)
            stateV = agent_list[agent_index].policy_old.actor.V_0
            state_v_list.append(stateV)
            # print(stateV)
            value = valuefunc(reward_list)
            stacked_val.append(value[0])
            stacked_val_array = np.array(stacked_val)
            if epoch % save_step == 0:
                np.savetxt('stacked_val.csv', stacked_val_array, delimiter=',')
            print('value: ',value[0])
            # print(V)
            
            if reward_list[-1] < -20:
                label = 0
            elif game_step == trajectory_step - 1:
                label = 1
            else:
                label = 2
            with open(information_name+'.pkl', 'ab+') as f:
                pickle.dump({"label": label, "state_expl_agent": state_expl_agent, "V":state_v_list}, f)
            # torch.save(agent_list[agent_index].policy_old.actor.state_dict(), information_name + '.pth')
            # plt.close(fig) 

            # for sampled_agent in agent_list:
            #     if len(reward_list) > 1:
            #         sampled_agent.update()
            #     if agent_index == 0:
            #         break
            
            # print(value, save_state_reward)
            # state_reward = (save_state_reward * torch.tensor(value).to(device))
            # loss = (state_reward.mean()) - torch.dot(prob_out_subtask.squeeze(), torch.log(prob_out_subtask.squeeze()))  # Calculate the loss
            # print('loss:', loss.item())
            # optimizer.zero_grad()
            # loss.backward()  # Backpropagate the loss
            # optimizer.step() 
            print('done')
            n_episodes += 1
            getout = create_getout_instance()
            for agent in agent_list:
                # agent.buffer.clear()
                agent.policy_old.actor.buffer.clear()
            # break
    # if (epoch+1) % update_steps == 0:
    #     for agent in agent_list:
    #         agent.update()

    print(reward_list)
    # exit()
# plt.imshow(image)
# plt.show(block=False)
# plt.pause(0.01)


# print("Extracted Features Shape:", features.shape)

exit()
print(pred_name_1 + pred_name)
# print('V:', V, 'V_1:', V_1, 'pred_name:', pred_name, 'pred_name_1:', pred_name_1)
state_img = getout.camera.screen
reward = getout.step(actual_action[0])
print(reward)
done = getout.level.terminated
if done:
    getout = create_getout_instance()
# print('state_expl:', state_expl, 'actual_action:', actual_action, 'done_list:', done_list, 'step_list:', state_img, 'reward_list:', reward_list, 'V:', V)

# ax.imshow(getout.camera.screen, cmap='gray')
plt.imshow(state_img)
plt.pause(0.01)
plt.show(block=False)
ax.clear()
