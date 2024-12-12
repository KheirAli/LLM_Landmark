import numpy as np
import torch
import os
import random
def read_and_classify(filename):
    with open(filename, 'r') as file:
        output_list = []
        logic_list = []
        output_value = []
        logic_value = []
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            
            if line.startswith('new data'):
                output_list.append(output_value)
                logic_list.append(logic_value)
                output_value = []
                logic_value = []
            elif not('.(__T__)' in line):
                output_value.append( line)
            else:
                print('LINE:', line)
                t =  (line.split('.(__T__)')[0])
                print('T IS HERE:', t)
                output_value.append(t.split(':')[1])
                print(output_value)
                logic_value.append( line.split('(__T__)')[1])
        print(output_list)
# Example usage:
def read_and_create_states(filename,list_of_predicartes):
    with open(filename, 'r') as file:
        output_list = []
        
        output_value = []
        logic_value = []
        for line in file:

            logic_list = []
            line = line.split(',')
            # print('LINE:', line)
            for idx, i in enumerate(line):
                if 'type' in i and 'type' in list_of_predicartes:
                    logic_list.append(i.strip()+','+line[idx+1].strip())
                if 'pickup' in i and 'pickup' in list_of_predicartes:
                    logic_list.append(i.strip())
                if 'reach' in i and 'reach' in list_of_predicartes:
                    logic_list.append(i.strip()+','+line[idx+1].strip())
                # if 'on_left' in i and 'on_left' in list_of_predicartes:
                #     logic_list.append(i.strip()+','+line[idx+1].strip())
                if 'closeby' in i and 'closeby' in list_of_predicartes:
                    logic_list.append(i.strip()+','+line[idx+1].strip())
            if len(logic_list)>0 or '(__T__)' in output_list[0]:
                output_list.append(logic_list)
        return output_list
                # 
# read_and_classify('output.txt')

states = read_and_create_states('output_correct.txt', list_of_predicartes = ['closeby', 'pickup', 'reach', 'type'])
# print('STATES:', (states))

typed_objects = set()
# new_states = []
# for state_1 in states:
#     for state in state_1:
#         if state.startswith('type('):
#             obj_type = state.split(',')[0].split('(')[1]
#             typed_objects.add(obj_type)

#     # Filter out 'closeby' relations where either object has no type
#     new_state = [state for state in state_1 if not state.startswith('closeby(') or 
#                 (state.split('(')[1].split(',')[0] in typed_objects and 
#                 state.split(',')[1].split(')')[0] in typed_objects)]
#     print('NEW STATE:', new_state, 'state_1:', state_1, 'typed_objects:', typed_objects)
#     # new_state = [state for state in state_1 if not state.startswith('on_left(') or 
#     #             (state.split('(')[1].split(',')[0] in typed_objects and 
#     #             state.split(',')[1].split(')')[0] in typed_objects)]
#     new_states.append(new_state)
# states = new_states
unique_states = []
for i in states:
    if i not in unique_states:
        unique_states.append(i)
print('UNIQUE STATES:', len(unique_states))
# print('UNIQUE STATES:', unique_states)
file_path = __file__
file_directory = os.path.dirname(file_path)
positive_set_state = torch.load(file_directory+'/PS.pth')
negative_set_state = torch.load(file_directory+'/NS.pth')
del positive_set_state[0]
del negative_set_state[0]
# print('POSITIVE SET STATE:', positive_set_state[0])
def check_list(list_1):
    if len(list_1) == 0:
        return None
    if len(list_1) == 1:
        if type(list_1[0]) == list:
            return list_1[0]
        else:
            return list_1
Candidate_state = []

for potential_state in unique_states:
    counter_list = []
    landmark = []
    for pos_traj in positive_set_state:
        counter = 0
        for pos_state in pos_traj:
            pos_state = check_list(pos_state)
            occurence_in_trajectory = 0
            for i in potential_state:
                if i in pos_state[0]:
                    occurence_in_trajectory+=1
                    
            # print('OCCURENCE IN TRAJECTORY:', occurence_in_trajectory, 'LEN:', len(potential_state))
            # print('POTENTIAL STATE:', potential_state)
            # print('POS STATE:', pos_state)
            
            if occurence_in_trajectory == len(potential_state):
                counter+=1
    
        if counter > 0:
            # print('POTENTIAL STATE:', potential_state)
            counter_list.append(counter)
            landmark.append(potential_state)
    # print('COUNTER LIST:', counter_list)
    if len(counter_list) == len(positive_set_state):
        # print('POTENTIAL STATE:', potential_state)
        print('COUNTER LIST:', min(counter_list),'similarity with uniform' ,np.mean(np.log(1/len(counter_list)*(counter_list/np.sum(counter_list))**(-1))))
        print('LANDMARK:', landmark[0])
        Candidate_state.append(landmark[0])
        print('-----------------------------------------')

print('CANDIDATE STATE:', Candidate_state)

for potential_state in Candidate_state:
    number_of_traj = 10
    off_set = random.randint(0, len(negative_set_state)-number_of_traj)
    counter_list = []
    landmark = []
    for pos_traj in negative_set_state[off_set:number_of_traj+off_set]:
        counter = 0
        for pos_state in pos_traj:
            pos_state = check_list(pos_state)
            occurence_in_trajectory = 0
            for i in potential_state:
                if i in pos_state[0]:
                    occurence_in_trajectory+=1
                    
            # print('OCCURENCE IN TRAJECTORY:', occurence_in_trajectory, 'LEN:', len(potential_state))
            # print('POTENTIAL STATE:', potential_state)
            # print('POS STATE:', pos_state)
            
            if occurence_in_trajectory == len(potential_state):
                counter+=1
    
        if counter > 0:
            # print('POTENTIAL STATE:', potential_state)
            counter_list.append(counter)
            landmark.append(potential_state)
    # print('COUNTER LIST:', counter_list)
    if len(counter_list) == number_of_traj:
        # print('POTENTIAL STATE:', potential_state)
        print('COUNTER LIST:', min(counter_list),'similarity with uniform' ,np.mean(np.log(1/len(counter_list)*(counter_list/np.sum(counter_list))**(-1))))
        print('LANDMARK:', landmark[0])
        
        print('-----------------------------------------')