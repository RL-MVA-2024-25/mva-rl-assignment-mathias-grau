from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeeperQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(DeeperQNetwork, self).__init__()
        
        # Shared feature trunk
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        # Common feature extraction
        return self.net(x)




class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(DuelingQNetwork, self).__init__()

        # Feature extraction 
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Value stream 
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream 
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)

        # Value and Advantage
        values = self.value_stream(features)              # shape: (batch_size, 1)
        advantages = self.advantage_stream(features)      # shape: (batch_size, action_dim)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values


class BiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class BBiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class BBBiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BBBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class BBBBiggerDuelingQNetwork(nn.Module): # le meilleur pour l'instant 
    def __init__(self, state_dim=6, action_dim=4):
        super(BBBBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
        )
        
        # Value stream256
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class BBBBBiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BBBBBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            # nn.Linear(1024, 1024), # best without
            # nn.SiLU(), # best without
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256), 
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
