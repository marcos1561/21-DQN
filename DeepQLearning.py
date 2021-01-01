import torch.nn as nn
import torch.optim as optim
import torch
import model_memory
import numpy as np
import random
import gym
import time
import os

NUM_ACOES = 2
NUM_ESTADOS = 3
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.01
EPSILON_DECAY = 0.9  
GAMMA = 0.999  
REPLAY_MEMORY_SIZE = 5000
MINIBATCH_SIZE = 256
STEPS_PER_EPISODE = 100
TOTAL_EPISODES = 50_000
LEARNING_RATE = 1e-5

# Dicionário de hiperparâmetros
hyperparams_dict = {'NUM_ACOES': NUM_ACOES,
                    'NUM_ESTADOS': NUM_ESTADOS,
                    'EPSILON_FINAL': EPSILON_FINAL,
                    'EPSILON_DECAY': EPSILON_DECAY,
                    'GAMMA': GAMMA,
                    'REPLAY_MEMORY_SIZE': REPLAY_MEMORY_SIZE,
                    'MINIBATCH_SIZE': MINIBATCH_SIZE,
                    'STEPS_PER_EPISODE': STEPS_PER_EPISODE,
                    'TOTAL_EPISODES': TOTAL_EPISODES,
                    'LEARNING_RATE': LEARNING_RATE,
                   }

env = gym.make("Blackjack-v0")
device = "cpu"

class DQNModel(nn.Module):
    def __init__(self, idel_sol, num_states=3, num_actions=2, weight_norm=False):
        
        super(DQNModel, self).__init__()

        # Inicializa os atributos do modelo
        self.current_episode = 0
        self.fc1 = nn.Linear(num_states, 96)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(96, 48)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(48, 24)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(24, num_actions)

        # Normaliza os pesos
        if weight_norm:
            self.fc1 = nn.utils.weight_norm(self.fc1)
            self.fc2 = nn.utils.weight_norm(self.fc2)
            self.fc3 = nn.utils.weight_norm(self.fc3)

        # Aplica a solução ideal
        # self.x_soft_hand = ideal_sol[0]
        # self.x_hard_hand = ideal_sol[1]
        # self.y_soft_hand = ideal_sol[2]
        # self.y_hard_hand = ideal_sol[3]

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)

        return out


def train_model(model, save_path, load_checkpoint, load_hyperparams):
    model.optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    memory = model_memory.Memory(REPLAY_MEMORY_SIZE)
    loss_func = nn.MSELoss()

    training_loss_logger = [0]
    hard_accuracy_logger = []
    soft_accuracy_logger = []
    model.loggers = training_loss_logger, hard_accuracy_logger, soft_accuracy_logger

    # Carrega Modelo
    checkpoint_loaded = load_model(model, save_path, load_checkpoint, load_hyperparams)

    # Treina
    if not checkpoint_loaded:
        logger = train_agent(model, model.optimizer, memory, loss_func, STEPS_PER_EPISODE, TOTAL_EPISODES, model.loggers, save_path)

    return model.loggers


def train_agent(model, optimizer, memory, loss_func, n_steps, n_episodes, loggers, save_path, exp_epsilon_decay=False):
    training_loss_logger, hard_accuracy_logger, soft_accuracy_logger = loggers

    epsilon_decrements = []

    if exp_epsilon_decay:
        epsilon_decrements = [EPSILON_INITIAL]
        
        found_eps_min = False
        for i in range(TOTAL_EPISODES):
            if epsilon_decrements[i] > EPSILON_FINAL:
                epsilon_decrements.append(epsilon_decrements[i] * EPSILON_DECAY)
            elif not found_eps_min:
                epsilon_decrements.append(epsilon_decrements[i])
                print(f'Valor Mínimo de Epsilon alcançado em {i} episódios')
                found_eps_min = True
            else:
                epsilon_decrements.append(epsilon_decrements[i])
   
    else:
        epsilon_decrements = np.linspace(EPSILON_INITIAL, EPSILON_FINAL, n_episodes+1)

    start_time = time.time()

    # Loop por cada episódio
    for episode_idx in range(model.current_episode, model.current_episode + n_episodes + 1):
        
        # Decremento do epsilon
        epsilon = epsilon_decrements[episode_idx - model.current_episode] 

        # Chamada à função de treinamento
        training_loss_logger = train_episode(model, optimizer, memory, loss_func, epsilon, n_steps, training_loss_logger)

        # hard_accuracy, soft_accuracy = avalia_modelo(model)
        # hard_accuracy_logger.append(hard_accuracy)
        # soft_accuracy_logger.append(soft_accuracy)
        loggers = training_loss_logger, hard_accuracy_logger, soft_accuracy_logger

        # Salva o modelo a cada 2000 episódios
        if episode_idx % 2000 == 0 or episode_idx == model.current_episode + n_episodes:
            torch.save({'episode_idx': episode_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loggers': loggers,
                        'hyperparams_dict': hyperparams_dict,
                        'episode_idx': episode_idx}, 
                       save_path)
            print('Modelo Salvo')
        
        if episode_idx % 200 == 0:
            print(f'| Episódio: {episode_idx:02} | Erro em Treinamento: {training_loss_logger[episode_idx-1]:0.2f}')

    end_time = time.time()
    print("Tempo de Total de Treinamento igual a %.2f segundos" % (end_time - start_time))

    return loggers


def avalia_modelo(model):
    return "noice"


def train_episode(model, optimizer, memory, loss_func, epsilon, n_sptes, loss_logger):
    cur_state = env.reset()

    with torch.no_grad():
        for step_i in range(n_sptes):
            cur_state_t = torch.Tensor(cur_state).to(device)
            action = choose_action(model, cur_state_t, epsilon)
            next_state, reward, is_done, _ = env.step(action)

            memory.save_memory(cur_state, action, reward, next_state, is_done)

            if is_done:
                cur_state = env.reset()
            else:
                cur_state = next_state

    if len(memory.memory) > MINIBATCH_SIZE:
        update_model(model, optimizer, loss_func, memory, loss_logger)

    return loss_logger


def choose_action(model, cur_state, epsilon):
    if random.random() < epsilon:
        return np.random.randint(0, NUM_ACOES)
    else:
        actions = model(cur_state) 
        values, index = actions.max(0)

    return index.item()


def update_model(model, optimizer, loss_func, memory, loss_logger):
    model.train()

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(MINIBATCH_SIZE)

    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    done_batch = done_batch.to(device)

    current_state_qs = model(state_batch) 
    next_state_qs = model(next_state_batch)
    
    target_qs_list = []

    # print(f"current_state: {state_batch[0]} \naction: {action_batch[0]} \nnext_state: {next_state_batch[0]} \nreward: {reward_batch[0]}\n")

    for i in range(MINIBATCH_SIZE):
        if not done_batch[i]:
            max_q_future = torch.max(next_state_qs[i])
            new_q = reward_batch[i] + GAMMA * max_q_future
        else:
            new_q = reward_batch[i]

        action_taken = int(action_batch[i].item())
        current_qs = current_state_qs[i].clone()
        current_qs[action_taken] = new_q
        target_qs_list.append(current_qs)
        
    target_qs_values = torch.stack(target_qs_list)

    optimizer.zero_grad()

    loss = loss_func(current_state_qs, target_qs_values)
    loss_logger.append(loss.item())
    loss.backward()

    optimizer.step()
    model.eval()


def load_model(model, save_path, load_chekpoint=False, load_hyperparams=False):
    if load_chekpoint:
        if os.path.isfile(save_path):
            check_point = torch.load(save_path)
            model.load_state_dict(check_point['model_state_dict'])
            model.optimizer.load_state_dict(check_point['optimizer_state_dict'])
            model.current_episode = check_point['episode_idx']
            model.loggers = check_point['loggers']
            print("Checkpoint Carregado. Iniciando do episódio:", model.current_episode)
        
            return True 
        else:
            print("Checkpoint não encontrado!")
            return False

    elif load_hyperparams:
        check_point = torch.load(save_path)
        hyperparams_dict = check_point['hyperparams_dict']

        global NUM_ACOES, NUM_ESTADOS, EPSILON_FINAL, EPSILON_FINAL, EPSILON_DECAY, GAMMA, \
               REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, STEPS_PER_EPISODE, TOTAL_EPISODES, LEARNING_RATE
        
        NUM_ACOES = hyperparams_dict['NUM_ACOES']
        NUM_ESTADOS = hyperparams_dict['NUM_ESTADOS']
        EPSILON_FINAL = ['EPSILON_FINAL']
        EPSILON_DECAY = hyperparams_dict['EPSILON_DECAY']
        GAMMA = hyperparams_dict['GAMMA']
        REPLAY_MEMORY_SIZE = hyperparams_dict['REPLAY_MEMORY_SIZE']
        MINIBATCH_SIZE = hyperparams_dict['MINIBATCH_SIZE']
        STEPS_PER_EPISODE = hyperparams_dict['STEPS_PER_EPISODE']
        TOTAL_EPISODES = hyperparams_dict['TOTAL_EPISODES']
        LEARNING_RATE = hyperparams_dict['LEARNING_RATE']

        print('Hiperparâmetros Carregados do Checkpoint!')
        return False

    else:
        print('Modelo Não Carregado!')
        return False
