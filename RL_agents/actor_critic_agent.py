import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

'''
    Agent has 2 networks: Actor and Critic
    Actor: Policy approximator -> tells the agent what action to take
           This is basically the policy(π), which is approximated by a neural network N with
           policy parameters Θ that we need to optimise to Θ* for optimal policy π* using the 
           gradient of the log prob or grad of log policy (d/dΘ(log π(Θ))):
           ΔΘ = alpha * d/dΘ(log π(Θ))
    Critic: Tells how much an action is good or bad. Look at the value [A(n) - b(n)] where A is actor and b is our critic
            defined as the expected sum of rewards(with discounting of γ per time step). But this is the same as the definition
            of our value function V(s)! Thus the critic is V(s). 
            And therefore, A(n) - b(n) becomes [R(t+1) + γ*V(S[n+1]) - V(S[n])], ie., the TD Error
'''
# This class is used by ActorCriticAgent which is used in main.py while running algo "actor_critic"
class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        
        #  Policy and critic value v
        self.pi = nn.Linear(self.fc4_dims, n_actions)
        self.v = nn.Linear(self.fc4_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = torch.device('cuda:0')
        self.to(self.device)

    def forward(self, state):
        state = torch.Tensor(state).to(self.device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        pi = self.pi(x)
        v = self.v(x)

        return pi, v

class ActorCriticAgent(object):
    def __init__(self, alpha, input_dims, 
                    n_actions,
                    gamma,
                    student_simulator,
                    layer1_size, 
                    layer2_size,
                    layer3_size,
                    layer4_size,
                    agent_type,
                    skill_group_to_activity_map=None,
                    skill_groups=None):
        
        self.agent_type = agent_type
        self.alpha = alpha    
        self.gamma = gamma
        self.epsilon = 0.0
        self.student_simulator = student_simulator
        
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size, layer2_size, 
                                                layer3_size, layer4_size, n_actions)
        # Value we use to update weights of NN, gradient log policy (since policy is probabilistic it is called gradient log probability dist. that is the policy)
        self.log_probs = None
        self.skill_groups = skill_groups
        self.skill_group_to_activity_map = skill_group_to_activity_map

    def choose_action(self, state, explore=False):
        
        skills, activityName = None, None
        policy, critic_value = self.actor_critic.forward(state)
        policy = F.softmax(policy, dim=0)  # softmax ensures actions add up to one which is a requirement for probabilities
        action_probs = torch.distributions.Categorical(policy)
        action = action_probs.sample()  # Sample an action from these proba's and get the log proba's.
        self.log_probs = action_probs.log_prob(action)

        student_model_name = self.student_simulator.student_model_name

        if self.agent_type == None:
            if student_model_name == 'ActivityBKT':
                # Assumption (needs to be changed): All activities within a skill group contribute to same amount 
                # and there is no difference between them in same skill group
                # So, we just sample a random activity under this skill group to present it to the student 
                # Possible Fix: Fit params per activity
                skills = self.skill_groups[action]
                skill_group = skills.copy()
                activityName = np.random.choice(self.skill_group_to_activity_map[str(action.item())])
        
        elif self.agent_type == 1:
            pass
        elif self.agent_type == 2:
            pass
        elif self.agent_type == 3 or self.agent_type == 4 or self.agent_type == 5:
            return action.item(), explore, None, None
        
        return action.item(), explore, skills, activityName

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)

        # TD ERROR
        delta = reward + (self.gamma * critic_value_ * (1-int(done))) - critic_value 

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()

        
# used only for PPO algo, Actor Critic Algo uses the class `ActorCriticNetwork`
class ActorCritic(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, agent_type=None):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.agent_type = agent_type
        self.lr = lr
        self.epochs = 0

        if agent_type == None or agent_type == 1 or agent_type == 2 or agent_type == 3 or agent_type == 5:       
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
            self.fc3 = nn.Linear(self.fc2_dims, self.fc2_dims)
            self.fc4 = nn.Linear(self.fc2_dims, self.fc2_dims)
            self.pi = nn.Linear(self.fc2_dims, n_actions)   #   Actor proposes policy; n_actions = 3, 1 for each threshold 
            self.v = nn.Linear(self.fc2_dims, 1)            #   Critic gives a value to criticise the proposed action/policy

        elif agent_type == 4:
            num_literacy_acts, num_math_acts, num_story_acts = n_actions[0], n_actions[1], n_actions[2]
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
            
            self.literacy_pi = nn.Linear(self.fc2_dims, num_literacy_acts)
            self.literacy_value = nn.Linear(self.fc2_dims, 1)

            self.math_pi = nn.Linear(self.fc2_dims, num_math_acts)
            self.math_value = nn.Linear(self.fc2_dims, 1)

            self.story_pi = nn.Linear(self.fc2_dims, num_story_acts)
            self.story_value = nn.Linear(self.fc2_dims, 1)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        decayRate = 0.97
        self.optimizer = optim.Adam(self.parameters(), lr = lr,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)
        self.to(self.device)

    def forward(self, state):
        self.epochs += 1

        if isinstance(state, list): state = torch.tensor(state)
        if torch.is_tensor(state) == False or isinstance(state, np.ndarray):    state = torch.FloatTensor(state)
        if state.get_device() != self.device:    state = state.to(self.device)

        if self.agent_type == None or self.agent_type == 5:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            pi = F.softmax(self.pi(x), dim=1)
            v = self.v(x)
            pi = torch.distributions.Categorical(pi)    # discrete actions
            return pi, v
        
        elif self.agent_type == 3:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            pi = F.softmax(self.pi(x), dim=1)
            v = self.v(x)
            pi = torch.distributions.Categorical(pi)    # discrete actions
            return pi, v

        elif self.agent_type == 1 or self.agent_type == 2:
            x = self.fc1(state)
            x = F.relu(x)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            value = self.v(x)
            probs = torch.sigmoid(self.pi(x))
            dist = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=probs)    # continuous actions in [0, 1]
            return dist, value
        
        elif self.agent_type == 4:
            pis = []
            values=  []
            matrix_nums = []

            for row in state.cpu().tolist():
                matrix_nums.append(int(row[-1]))

            for i in range(len(state)):
                row = state[i].view(1, -1)
                x = self.fc1(row)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.relu(x)

                if matrix_nums[i] == 1:
                    literacy_pi = F.softmax(self.literacy_pi(x), dim=1)
                    literacy_value = self.literacy_value(x).view(-1)
                    literacy_pi = torch.distributions.Categorical(literacy_pi)
                    pis.append(literacy_pi)
                    values.append(literacy_value)

                elif matrix_nums[i] == 2:
                    math_pi = F.softmax(self.math_pi(x), dim=1)
                    math_value = self.math_value(x).view(-1)
                    math_pi = torch.distributions.Categorical(math_pi)
                    pis.append(math_pi)
                    values.append(math_value)

                elif matrix_nums[i] == 3:
                    story_pi = F.softmax(self.story_pi(x), dim=1)
                    story_value = self.story_value(x).view(-1)
                    story_pi = torch.distributions.Categorical(story_pi)
                    pis.append(story_pi)
                    values.append(story_value)

            if len(values) != 0:    values = torch.stack(values)
            
            return pis, values