import gymnasium as gym
from stable_baselines3 import PPO, DQN
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio

ENV_NAME = "LunarLander-v3"
TOTAL_TIMESTEPS = 100000 
PROJECT_NAME = "nn2526_projekt3"

def train_model(
        algo_class, 
        algo_name: str, 
        run_name: str, 
        policy: str = "MlpPolicy", # spróbować też np. "CnnPolicy" 
        timesteps: int  = TOTAL_TIMESTEPS, 
        env_name: str = ENV_NAME # spróbować też innych środowisk (?)
    ):

    print(f"\n{'='*40}")
    print(f"Training {algo_name}")
    print(f"{'='*40}")

    run = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        config={"algo": algo_name, "env": env_name, "timesteps": timesteps},
        sync_tensorboard=True, 
        reinit=True
    )

    env = gym.make(env_name) # można pozmieniać grawitacje, wiatr, turbulencje 

    # rozważyć tuning również innych parametrów 
    # np. 
    # (dla ppo) gae_lambda, entropy_coefficient, value-function_coefficient
    # (dla dqn) buffer_size, exploration_frac, exploration_init_epsilon, exploration_final_epsilon
    lr = 1e-3
    batch_size = 32
    gamma = 0.99

    model = algo_class(
        policy,
        env,
        verbose=0,
        tensorboard_log=f"runs/{run_name}",
        learning_rate = lr,
        batch_size = batch_size,
        gamma = gamma
    ) # można dodać też lr scheduler tutaj 

    model.learn(
        total_timesteps=timesteps,
        callback=WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run_name}",
            verbose=0
        )
    )

    model_path = f"models/{algo_name.lower()}_{run_name}"
    model.save(model_path)
    
    env.close()
    run.finish()

    return model_path

def record(model_path, algo_class, filename, seconds=10, env_name = ENV_NAME):
    print(f"\n--- Recording {seconds}s video for {algo_class.__name__} ---")

    model = algo_class.load(model_path)

    env = gym.make(env_name, render_mode="rgb_array")

    # LunarLander-v3 usually runs at 50 FPS
    fps = env.metadata.get("render_fps", 50)
    total_frames = fps * seconds

    frames = []
    obs, _ = env.reset()

    for _ in range(total_frames):
        frames.append(env.render())
        
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    imageio.mimsave(filename, frames, fps=fps)
    print(f"Successfully saved video to: {filename}")

if __name__ == "__main__":
    ppo_path = train_model(PPO, "PPO", "test_ppo_1")
    record(ppo_path, PPO, "logs/videos/ppo_lunarlander_10s.mp4")

    dqn_path = train_model(DQN, "DQN", "test_dqn_1")
    record(dqn_path, DQN, "logs/videos/dqn_lunarlander_10s.mp4")

    # zaimplementować też zwykły reinforce (+ double dqn ??)
    # odpalić na różnych seedach i uśrednić wyniki (?)