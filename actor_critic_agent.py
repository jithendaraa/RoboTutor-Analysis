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

class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        
        self.n_actions = n_actions
        self.lr = lr
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0')
        self.to(self.device)

    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

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
        action_probs = torch.distributions.Categorical(policy)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        
        # skills_associated = self.activity_to_skills_map[self.cta_tutor_ids.tolist()[action]]
        # skills = []

        # for skill in skills_associated:
        #     skills.append(self.skill_to_number_map[skill])

        skills = self.skill_groups[action]
        skill_group = skills.copy()

        activity = np.random.choice(self.skill_group_to_activity_map[str(action.item())])

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

        
    

# class ActorCriticAgent(object):
#     def __init__(self, alpha, beta, input_dims, gamma=0.99, l1_size=512, l2_size=1024, l3_size=1024, n_actions=1710):
#         self.gamma = gamma
#         self.log_probs = None
#         self.actor = GenericNetwork(alpha, input_dims, l1_size, l2_size, l3_size, 
#                                     n_actions)
#         self.critic = GenericNetwork(beta, input_dims, l1_size=256, l2_size=256, l3_size=256, 
#                                     n_actions=1)

#     def choose_action(self, state):
#         # softmax ensures actions add up to one which is a requirement for probabilities
#         probabilities = F.softmax(self.actor.forward(state))

#         # create a distribution that is modelled on these probabilites
#         action_probs = torch.distributions.Categorical(probabilities)
#         # Sample from action_probs
#         action = action_probs.sample()
#         self.log_probs = action_probs.log_prob(action)
        
#         # action is a tensor, but we need action as int
#         return action.item()
    
#     def learn(self, state, reward, new_state, done):
#         self.actor.optimizer.zero_grad()
#         self.critic.optimizer.zero_grad()

#         critic_value = self.critic.forward(state)
#         critic_value_ = self.critic.forward(new_state)

#         # TD Error: [R(t+1) + γ*V(S[n+1]) - V(S[n])]
#         delta = ((reward + self.gamma*critic_value_*(1-int(done))) - critic_value)

#         # Maximize self.log_probs * delta, greatest possible future reward
#         actor_loss = -self.log_probs * delta
#         critic_loss = (delta)**2 

#         (actor_loss + critic_loss).backward()

#         self.actor.optimizer.step()
#         self.critic.optimizer.step()