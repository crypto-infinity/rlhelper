# Reinforcement Learning Helper

`rlhelper` è una utility Python per Reinforcement Learning che fornisce implementazioni pronte all'uso di algoritmi tabulari e deep RL per ambienti OpenAI Gym/Gymnasium.

## Funzionalità principali
- Q-Learning tabulare
- SARSA tabulare
- Dyna-Q (tabulare con planning)
- Double Deep Q-Network (DDQN) con PyTorch

## Compatibilità e Note

- rlhelper richiede **Gymnasium >=0.28** (non è compatibile con OpenAI Gym classico).
- La funzione `dyna_q` utilizza `env.current_action_space`, che potrebbe non essere presente in tutti gli ambienti standard Gymnasium.
- Tutte le funzioni sono pensate per ambienti con spazi di osservazione e azione discreti (tranne DDQN, che supporta anche osservazioni vettoriali).
- Il codice è testato con Python >=3.8.

## Installazione

Installa le dipendenze richieste con:

```bash
pip install -r requirements.txt
```

oppure manualmente:

```bash
pip install torch numpy gymnasium
```

## Utilizzo

### Q-Learning
```python
import gymnasium as gym
from rlhelper import RLHelper

env = gym.make('FrozenLake-v1', is_slippery=False)
Q, policy = RLHelper.q_learning(env, episodes=5000)
print('Q-table:', Q)
print('Policy:', policy)
```

### SARSA
```python
import gymnasium as gym
from rlhelper import RLHelper

env = gym.make('FrozenLake-v1', is_slippery=False)
Q, policy = RLHelper.sarsa(env, episodes=5000)
print('Q-table:', Q)
print('Policy:', policy)
```

### Dyna-Q
```python
import gymnasium as gym
from rlhelper import RLHelper

env = gym.make('FrozenLake-v1', is_slippery=False)
Q, model, policy = RLHelper.dyna_q(env, episodes=1000, planning=5)
print('Q-table:', Q)
print('Policy:', policy)
```

### DDQN (Deep Q-Network)
```python
import gymnasium as gym
import torch.nn as nn
from rlhelper import RLHelper

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

env = gym.make('CartPole-v1')
rewards = RLHelper.ddqn(env, DQN, episodes=200)
print('Rewards per episode:', rewards)
```

## API Principali

- `q_learning(env, ...)` — Q-Learning tabulare
- `sarsa(env, ...)` — SARSA tabulare
- `dyna_q(env, ...)` — Dyna-Q tabulare
- `ddqn(env, dqn_class, ...)` — Double Deep Q-Network (PyTorch)

Consulta le docstring nel codice per la descrizione dettagliata dei parametri.

## Licenza

Questo progetto è distribuito sotto licenza Apache 2.0. Vedi il file LICENSE per i dettagli.