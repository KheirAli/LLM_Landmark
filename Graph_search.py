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
                if 'on_left' in i and 'on_left' in list_of_predicartes:
                    logic_list.append(i.strip()+','+line[idx+1].strip())
                if 'closeby' in i and 'closeby' in list_of_predicartes:
                    logic_list.append(i.strip()+','+line[idx+1].strip())
            if len(logic_list)>0 or '(__T__)' in output_list[0]:
                output_list.append(logic_list)
        return output_list
                # 
# read_and_classify('output.txt')

from itertools import chain, combinations

# List of predicates
list_of_predicates = ['closeby', 'pickup', 'reach', 'type', 'on_left']

# Function to generate all subsets of the list
def find_all_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(len(lst) + 1)))

# Find all subsets
subsets = find_all_subsets(list_of_predicates)
predicates_nodes = []
for i in subsets:
   if len(i) > 0:
    predicates_nodes.append(list(i))

# Display subsets to the user
# print(subsets)
predicates_nodes.reverse()
predicates_numb = len(predicates_nodes[0])
all_states_numb = 0
all_node_states = []
priority = []
for frontir_node_predicate in predicates_nodes:

    states = read_and_create_states('output_correct.txt', list_of_predicartes = frontir_node_predicate)
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
    # print('UNIQUE STATES:', len(unique_states))
    if len(frontir_node_predicate) == predicates_numb:
        all_states_numb = len(unique_states)
    # print('UNIQUE STATES:', unique_states)
    all_node_states.append(unique_states)
    priority.append(-len(unique_states)/all_states_numb+(len(frontir_node_predicate)-predicates_numb)+1)

length_to_indices = {}
priority = np.array(priority)

for index, sublist in enumerate(predicates_nodes):
    length = len(sublist)
    if length not in length_to_indices:
        length_to_indices[length] = []
    length_to_indices[length].append(index)

# Convert dictionary to list of tuples
result = [(length, indices) for length, indices in length_to_indices.items()]
def softmax(x, temperature=0.1):
    x = np.array(x)  # Ensure x is a NumPy array
    exp_x = np.exp(x / temperature)  # Scale by temperature
    return exp_x / np.sum(exp_x)

# Random sampling function
def random_sampling(size):
    return np.random.rand(size)
sampling_order_all = []
for length, indices in result:
    random_scores = random_sampling(len(indices))
    
    # print(indices, priority)
    probabilities = softmax(priority[indices])
    # print(probabilities)
    sampling_order = np.random.choice(indices, size=len(indices), replace=False, p=probabilities)
    for i in sampling_order :
        sampling_order_all.append((i))
    # print(sampling_order)

# print(sampling_order_all)
all_node_states = [all_node_states[i] for i in sampling_order_all]

for unique_states in all_node_states:
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
    landmark = []
    for potential_state in unique_states:
        counter_list = []
        
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


            # print('LANDMARK:', landmark[0])
            Candidate_state.append(landmark[-1])


    # print('CANDIDATE STATE:', Candidate_state)
    # print('Landmark:', landmark)
    final_candidate_state = Candidate_state.copy()
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
            #     # print('POTENTIAL STATE:', potential_state)
                counter_list.append(counter)
            #     landmark.append(potential_state)
        # print('COUNTER LIST:', counter_list)
        if len(counter_list) == number_of_traj:
            # print('POTENTIAL STATE:', potential_state)
            final_candidate_state.remove(potential_state)

    print("Landmark:", final_candidate_state)
    if len(final_candidate_state) >0:
        print("Done")
        break