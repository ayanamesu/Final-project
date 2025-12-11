import random
# from rsaenv import RSAEnv

# env = RSAEnv(capacity=20, data_dir="../data/train")

# obs, info = env.reset()
# print("Initial observation:", obs.shape)

# done = False
# step = 0

# while not done and step < 5:
#     print("\n=== STEP", step, "===")
#     # Try all actions (0–7) and pick one that is valid
#     # Find valid actions for current request
#     req = env.current_request
#     valid_actions = []

#     for a in range(8):
#         if env._map_action_to_path(req.source, req.destination, a) is not None:
#             valid_actions.append(a)

#     # Choose a valid action
#     action = random.choice(valid_actions)


#     obs, reward, terminated, truncated, info = env.step(action)

#     print("Action:", action)
#     print("Reward:", reward)
#     print("Blocked / Successful:", info)
#     print("Obs shape:", obs.shape)

#     step += 1
#     done = terminated

# print("\nEnvironment test finished.")

from rsaenv import RSAEnv

env = RSAEnv(capacity=20, data_dir="../data/train")
obs, _ = env.reset()

print("Initial observation:", obs.shape)

print("\n=== FORCING BLOCKING ===")
for step in range(10):
    # choose an invalid action intentionally
    action = 7  
    obs, reward, term, trunc, info = env.step(action)

    print(f"\n=== STEP {step} ===")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Blocked / Successful: {info}")
    print("Obs shape:", obs.shape)

    if term:
        break
