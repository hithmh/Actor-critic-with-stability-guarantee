import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['pong']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}NoFrameskip-v5'.format(name),
            entry_point='ENV.env.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

#classic
register(
    id='CartPolecons-v0',
    entry_point='ENV.env.classic_control:CartPoleEnv_cons',
    max_episode_steps=2500,
)

register(
    id='CartPolecost-v0',
    entry_point='ENV.env.classic_control:CartPoleEnv_cost',
    max_episode_steps=2500,
)

register(
    id='Carcost-v0',
    entry_point='ENV.env.classic_control:CarEnv',
    max_episode_steps=600,
)
# mujoco

register(
    id='HalfCheetahcons-v0',
    entry_point='ENV.env.mujoco:HalfCheetahEnv_lya',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id='HalfCheetahcost-v0',
    entry_point='ENV.env.mujoco:HalfCheetahEnv_cost',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Pointcircle-v0',
    entry_point='ENV.env.mujoco:PointEnv',
    max_episode_steps=65,
)

register(
    id='Antcons-v0',
    entry_point='ENV.env.mujoco:AntEnv_cpo',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Antcost-v0',
    entry_point='ENV.env.mujoco:AntEnv_cost',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Quadrotorcons-v0',
    entry_point='ENV.env.mujoco:QuadrotorEnv',
    max_episode_steps=512,
)



