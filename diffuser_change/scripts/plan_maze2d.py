import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-dense-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# 扩散planner的调用接口
policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation = env.reset()

if args.conditional:
    print('Resetting target')
    env.set_target()

## 设置目标位置，让扩散模型知道第0步在起点，第H-1步在终点
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
# 用规划的轨迹当 waypoint 控制，顺便可视化
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## 在 t=0 时，做一次“调用扩散 planner”的动作：
    if t == 0:
        cond[0] = observation # 告诉扩散模型起点观测

        action, samples = policy(cond, batch_size=args.batch_size)
        actions = samples.actions[0]
        sequence = samples.observations[0]
    # pdb.set_trace()

    '''
    如果当前时间步 t 还在轨迹长度范围内，就取下一时刻的预测状态 sequence[t+1] 作为当前的目标 waypoint；
	如果已经超过轨迹长度，就一直盯着最后一个 waypoint，并把它的速度部分设为 0（不再移动）。
    '''
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
        # pdb.set_trace()

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()


    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    ## 如果是 Maze2D，还额外打印当前位置和目标：
    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## 把 next_observation 记到 rollout 中，用于后面画 rollouts：
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)
    
    '''
        	1.	在 t=0 时，用 renderer.composite 把“扩散采样得到的那些轨迹样本 samples.observations”画出来，保存成 0.png。
            •	对你来说，这就是“planner 在静态地图上设计的 plan 样本们”。
            2.	每隔 vis_freq 步，或者 episode 结束时，把目前实际执行的 rollout 画成一张 rollout.png。
            •	这是“agent 真正走出来的路径”。
    '''
    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)
print("logbase:", args.logbase)
print("dataset:", args.dataset)
print("diffusion_loadpath:", args.diffusion_loadpath)
print("diffusion_epoch(arg):", args.diffusion_epoch)
print("diffusion_epoch(loaded):", diffusion_experiment.epoch)
print("savepath:", args.savepath)
print("model class:", type(diffusion_experiment.ema))

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
