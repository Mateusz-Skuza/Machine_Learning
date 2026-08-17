from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio
import os

ENV_NAME = "Acrobot-v1"
TOTAL_TIMESTEPS = 100000
PROJECT_NAME = "nn2526_projekt3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# WYMAGANA POPRAWKA DLA WANDB PRZY KILKU RUNACH W JEDNYM SKRYPCIE:
wandb.tensorboard.patch(root_logdir=os.path.join(BASE_DIR, "runs"))

MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "logs", "videos")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def train_model(
        algo_class,
        algo_name: str,
        run_name: str,
        policy: str = "MlpPolicy",
        timesteps: int = TOTAL_TIMESTEPS,
        env_name: str = ENV_NAME,
        lr: float = 1e-4,
        batch_size: int = 64,

        # DQN
        gamma: float = 0.99,
        buffer_size: int = 100000,
        exploration_fraction: float = 0.3,

        # PPO
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0
):

    print(f"\n{'='*40}")
    print(f"Training {algo_name} | Run: {run_name}")
    print(f"{'='*40}")

    run = wandb.init(
        project=PROJECT_NAME,
        entity="nn2526",
        name=run_name,
        config={
            "algo": algo_name,
            "env": env_name,
            "timesteps": timesteps,
            "learning_rate": lr,
            "batch_size": batch_size,
            "gamma": gamma,
            "buffer_size": buffer_size,
            "exploration_fraction": exploration_fraction,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef
        },
        sync_tensorboard=True,
        reinit=True
    )

    env = gym.make(env_name)

    custom_policy_kwargs = dict(net_arch=[256, 256])

    if algo_name == "QRDQN":
        model = algo_class(
            policy,
            env,
            verbose=0,
            tensorboard_log=os.path.join(BASE_DIR, "runs", run_name),
            learning_rate=linear_schedule(lr),
            batch_size=batch_size,
            gamma=gamma,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
            policy_kwargs=custom_policy_kwargs,
            target_update_interval=250,
            train_freq=4
        )

    elif algo_name == "DQN":
        model = algo_class(
            policy,
            env,
            verbose=0,
            tensorboard_log=os.path.join(BASE_DIR, "runs", run_name),
            learning_rate=linear_schedule(lr),
            batch_size=batch_size,
            gamma=gamma,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
            policy_kwargs=custom_policy_kwargs,
            target_update_interval=250,
            train_freq=4
        )

    elif algo_name == "PPO":
        model = algo_class(
            policy,
            env,
            verbose=0,
            tensorboard_log=os.path.join(BASE_DIR, "runs", run_name),
            learning_rate=linear_schedule(lr),
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            policy_kwargs=custom_policy_kwargs
        )

    model.learn(
        total_timesteps=timesteps,
        callback=WandbCallback(
            gradient_save_freq=1000,
            model_save_path=os.path.join(MODELS_DIR, run_name),
            verbose=0
        )
    )

    model_path = os.path.join(
        MODELS_DIR,
        f"{algo_name.lower()}_{run_name}"
    )

    model.save(model_path)

    env.close()
    run.finish()

    return model_path


def record(model_path, algo_class, filename, seconds=20, env_name=ENV_NAME):

    print(f"\n--- Recording {seconds}s video for {algo_class.__name__} ---")

    model = algo_class.load(model_path)
    env = gym.make(env_name, render_mode="rgb_array")

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

    
    ppo_base = train_model(PPO, "PPO", "ppo_acrobot_base", lr=1e-3, batch_size=64, gamma=0.99, ent_coef=0.0, gae_lambda=0.95)
    record(ppo_base, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_base.mp4"), seconds=20)

    dqn_base = train_model(DQN, "DQN", "dqn_acrobot_base", lr=5e-4, batch_size=64, gamma=0.99, buffer_size=100000, exploration_fraction=0.3)
    record(dqn_base, DQN, os.path.join(VIDEOS_DIR, "dqn_acrobot_base.mp4"), seconds=20)

  
    ppo_lr_low = train_model(PPO, "PPO", "ppo_acrobot_lr_1e-4", lr=1e-4)
    record(ppo_lr_low, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_lr_1e-4.mp4"), seconds=20)

    ppo_lr_high = train_model(PPO, "PPO", "ppo_acrobot_lr_5e-3", lr=5e-3)
    record(ppo_lr_high, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_lr_5e-3.mp4"), seconds=20)

    
    ppo_batch_32 = train_model(PPO, "PPO", "ppo_acrobot_batch_32", batch_size=32)
    record(ppo_batch_32, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_batch_32.mp4"), seconds=20)

    ppo_batch_256 = train_model(PPO, "PPO", "ppo_acrobot_batch_256", batch_size=256)
    record(ppo_batch_256, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_batch_256.mp4"), seconds=20)

    ppo_gamma_90 = train_model(PPO, "PPO", "ppo_acrobot_gamma_0.90", gamma=0.90)
    record(ppo_gamma_90, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_gamma_0.90.mp4"), seconds=20)

    ppo_ent_05 = train_model(PPO, "PPO", "ppo_acrobot_ent_0.05", ent_coef=0.05)
    record(ppo_ent_05, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_ent_0.05.mp4"), seconds=20)

    ppo_gae_80 = train_model(PPO, "PPO", "ppo_acrobot_gae_0.8", gae_lambda=0.8)
    record(ppo_gae_80, PPO, os.path.join(VIDEOS_DIR, "ppo_acrobot_gae_0.8.mp4"), seconds=20)

  
    dqn_lr_low = train_model(DQN, "DQN", "dqn_acrobot_lr_1e-4", lr=1e-4)
    record(dqn_lr_low, DQN, os.path.join(VIDEOS_DIR, "dqn_acrobot_lr_1e-4.mp4"), seconds=20)

    dqn_buf_1k = train_model(DQN, "DQN", "dqn_acrobot_buffer_1000", buffer_size=1000)
    record(dqn_buf_1k, DQN, os.path.join(VIDEOS_DIR, "dqn_acrobot_buffer_1000.mp4"), seconds=20)

    dqn_buf_10k = train_model(DQN, "DQN", "dqn_acrobot_buffer_10000", buffer_size=10000)
    record(dqn_buf_10k, DQN, os.path.join(VIDEOS_DIR, "dqn_acrobot_buffer_10000.mp4"), seconds=20)

    dqn_exp_10 = train_model(DQN, "DQN", "dqn_acrobot_exp_0.1", exploration_fraction=0.1)
    record(dqn_exp_10, DQN, os.path.join(VIDEOS_DIR, "dqn_acrobot_exp_0.1.mp4"), seconds=20)

    dqn_exp_50 = train_model(DQN, "DQN", "dqn_acrobot_exp_0.5", exploration_fraction=0.5)
    record(dqn_exp_50, DQN, os.path.join(VIDEOS_DIR, "dqn_acrobot_exp_0.5.mp4"), seconds=20)

   
    dqn_batch_128 = train_model(DQN, "DQN", "dqn_acrobot_batch_128", batch_size=128)
    record(dqn_batch_128, DQN, os.path.join(VIDEOS_DIR, "dqn_acrobot_batch_128.mp4"), seconds=20)

    print("\n--- WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE ---")