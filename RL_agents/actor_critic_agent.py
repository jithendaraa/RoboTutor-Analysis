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
                    activity_to_skills_map, 
                    kc_to_tutorID_dict, 
                    cta_tutor_ids, 
                    kc_list,
                    skill_to_number_map,
                    skill_groups, 
                    skill_group_to_activity_map,
                    gamma,
                    layer1_size, 
                    layer2_size,
                    layer3_size,
                    layer4_size,
                    n_actions
                    ):
        self.alpha = alpha    
        self.activity_to_skills_map = activity_to_skills_map
        self.kc_to_tutorID_dict = kc_to_tutorID_dict
        self.cta_tutor_ids = cta_tutor_ids
        self.kc_list = kc_list
        self.skill_to_number_map = skill_to_number_map
        self.gamma = gamma
        self.epsilon = 0.0
        
        self.actor_critic = ActorCriticNetwork(alpha, input_dims,
                                                layer1_size, layer2_size, layer3_size, layer4_size,
                                                n_actions=n_actions)
        # Value we use to update weights of NN, gradient log policy (since policy is probabilistic it is called gradient log probability dist. that is the policy)
        self.log_probs = None
        self.skill_groups = skill_groups
        self.skill_group_to_activity_map = skill_group_to_activity_map
    
    def choose_action(self, state, explore=False):
        
        policy, critic_value = self.actor_critic.forward(state)
        # softmax ensures actions add up to one which is a requirement for probabilities
        policy = F.softmax(policy)

        # Sample an action from these proba's and get the log proba's.
        action_probs = torch.distributions.Categorical(policy)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        
        skills = self.skill_groups[action]
        skill_group = skills.copy()

        # Assumption (needs to be changed): All activities within a skill group contribute to same amount and there is no difference between them in same skill group
        # So, we just sample a random activity under this skill group to present it to the student 
        activity = np.random.choice(self.skill_group_to_activity_map[str(action.item())])

        # action is a tensor. Hence we return action.item() which is just the value
        return action.item(), explore, skills, activity

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

        
# used only in ppo_agent.py for PPO algo, Actor Critic Algo uses the class `ActorCriticNetwork`
class ActorCritic(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, n_actions):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        #  Policy and critic value v
        self.pi = nn.Linear(self.fc1_dims, n_actions)
        self.v = nn.Linear(self.fc1_dims, 1)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.to(self.device)

    def forward(self, state):
        # state should be a torch.Tensor
        x = F.relu(self.fc1(state))
        pi = self.pi(x)
        v = self.v(x)
        return pi, v
