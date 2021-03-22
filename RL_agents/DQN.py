import pandas as pd
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, agent_type=None):
        super(DeepQNetwork, self).__init__()
        self.agent_type = agent_type
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0')
        self.to(self.device)

    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        if self.agent_type == None or self.agent_type == 3:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            actions = self.fc4(x)
        
        elif self.agent_type == 2:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            actions = F.softmax(self.fc4(x), dim=0)

        return actions

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, agent_type=None, activity_to_skills_map=None, kc_to_tutorID_dict=None, cta_tutor_ids=None, kc_list=None,
                max_mem_size=1000000, eps_min=0.05, eps_dec=0.998):
        
        print("Initialising DQN Agent type(" + str(agent_type) + ")")
        self.agent_type = agent_type
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr 
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.activity_to_skills_map = activity_to_skills_map
        self.kc_to_tutorID_dict = kc_to_tutorID_dict
        self.cta_tutor_ids = cta_tutor_ids
        self.num_skills = 18
        self.kc_list = kc_list
        
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, fc3_dims=256, n_actions=self.n_actions, agent_type=self.agent_type)
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1. - terminal

        if self.agent_type == None or self.agent_type == 3 or self.agent_type == 4 or self.agent_type == 5:
            actions = np.zeros(self.n_actions)
            actions[action] = 1.0 # one hot encoded actions

        elif self.agent_type == 1:
            pass
        elif self.agent_type == 2:
            pass
        # 0, 1, 2, 3.. representation to a one-hot encoded version; index int actions one-hot (for self.type discrete action agent type)
        self.action_memory[index] = actions
        self.mem_cntr += 1 
    
    def choose_action(self, state, explore=False):
        agent_type = self.agent_type
        rand = np.random.random()
        skill = None

        if agent_type is None:
            # explore
            if rand < self.epsilon and explore == True:
                sample_skill = np.random.choice(self.num_skills)    # an activity which exercises this skill will be sampled; since skill to num_actiivties that exercise this skills are skewed; skill[17] has count 2 but skill[0] has count > 600 and exploring maybe be biased to skills with a high count; so we choose skill number first and sample an activity that exercises the chosen skill
                # kc_to_tutorID_dict[kc_list[sample_skill]] -> activities that exercise skill `sample_skill`
                activity = random.choice(self.kc_to_tutorID_dict[self.kc_list[sample_skill]])
                action = self.cta_tutor_ids.tolist().index(activity)
            else: 
                actions = self.Q_eval.forward(state)
                action = torch.argmax(actions).item()
        
        elif agent_type == 1:
            pass

        elif agent_type == 2:
            if rand < self.epsilon and explore == True:
                action = np.array([random.random() for i in range(self.n_actions)])
            else: 
                action = self.Q_eval.forward(state).detach().cpu().numpy()
        
        elif agent_type == 3:
            if rand < self.epsilon and explore == True:
                action = np.random.choice(self.action_space)
            else: 
                actions = self.Q_eval.forward(state)
                action = torch.argmax(actions).item()

        elif agent_type == 4:
            pass

        elif agent_type == 5:
            pass

        return action, explore, skill
    
    def learn(self, decrease_eps = True):
        # Only learn when there is enough memory to fill up the whole batch
        if self.mem_cntr > self.batch_size: 
            self.Q_eval.optimizer.zero_grad()   # Zero out the gradients so we don't get exploding gradients

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                        else self.mem_size
            
            batch = np.random.choice(max_mem, self.batch_size)

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.uint8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.new_state_memory[batch]

            reward_batch = torch.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = torch.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            # Update Q values for max. actions
            _, max_actions = torch.max(q_next, dim=1)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            action_indices = action_indices.tolist()

            q_target[batch_index, action_indices] = reward_batch + \
                        self.gamma * torch.max(q_next, dim=1)[0] * terminal_batch
            
            # decrement epsilon if epsilon > eps_min
            if decrease_eps == True:
                self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_min \
                            else self.eps_min
            
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            


