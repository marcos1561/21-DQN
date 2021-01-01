import DeepQLearning as dql

model = dql.DQNModel(idel_sol=None).to("cpu")

# loggers = dql.train_model(model)

### JOGAR

env = dql.env
torch = dql.torch
device = "cpu"
1
continuar = True
while continuar:
    num_episodes = 50000
    total_wins = 0
    for _ in range(num_episodes):
        cur_state = env.reset()

        is_done = False
        while not is_done:
            with torch.no_grad():
                cur_state_t = torch.Tensor(cur_state).to(device)
                
                actions = model(cur_state_t) 
                values, index = actions.max(0)
                action = index.item()

                next_state, reward, is_done, _ = env.step(action)

                if is_done:
                    if reward > 0:
                        total_wins += reward
                else:
                    cur_state = next_state

    print(round(total_wins/num_episodes*100, 4))
    continuar = int(input("continuar? "))
