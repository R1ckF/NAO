
import roboschool
import gym
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from networkClassFC import *
from gymClass import *
import time
import os
import logging
import pickle

"""
Main function used for running each training run
Takes the variable parameters that are tested as inputs
"""
def main(play=True, nsteps=2048, loadPath=None, clippingFactor=lambda f: 0.2, epochs=10,
        nMiniBatch=4, learningRate=lambda f: f * 3.0e-4, activation=tf.nn.tanh, numNodes=64,
         numLayers=2, seed=0, loglevel=logging.DEBUG, checkpoint=None):
    ##define logger for printing
    LOGGER = logging.getLogger()
    logging.basicConfig(format='%(message)s', level=loglevel)

    ##define some constants similar for each training run
    saveInterval = 50000
    envName = "Hopper-v2"
    logInterval = 50000
    numSteps = 300000
    Lamda = 0.95
    gamma = 0.99
    networkStyle='copy'
    c1 = 0.5
    loadPath = loadPath
    checkpoint= checkpoint
    monitor=250
    if loadPath: numSteps=0
    loadModel = os.path.join(loadPath, checkpoint) if loadPath else None
    resultsPath = os.path.join("results",envName+str(numNodes)+"_"+str(numLayers)+"_tanh")

    network_args = {'networkOption': 'fc'}
    for item in ['activation', 'epochs', 'nMiniBatch','loadModel','networkStyle', 'c1', 'numNodes', 'numLayers']:
        network_args[item]=locals()[item]
    # network_args['kernel_initializer'] = tf.ones_initializer()
    LOGGER.debug(network_args)
    ## ensure values match to avoid errors later on
    assert ((nsteps/nMiniBatch) % 1 == 0)

    #create environement
    def buildEnv(envName, monitoring=False, normalize=False):
        env = gym.make(envName)
        env.NORMALIZED=False
        env.MONITOR = False
        if monitoring:
            env = gym.wrappers.Monitor(env, os.path.join(resultsPath,'videos'), video_callable=lambda episode_id: episode_id%monitoring==0, force=True)
            env.MONITOR=True
        if normalize:
            env = Normalize(env)
            env.NORMALIZED=True
        LOGGER.info('Creating env.. monitor: %s, normalize: %s', monitoring, normalize)
        return env, normalize, monitoring

    # def reloadEnv(loadedEnv, createdEnv): #This function allows for both replay and monitor to be activated on the same #!/usr/bin/env python
    #     if createdEnv.MONITOR:
    #         if createdEnv.NORMALIZED:
    #             createdEnv = createdEnv.env
    #     else:
    #         if createdEnv.N
    env, NORMALIZED, _ = buildEnv(envName, monitoring=monitor, normalize=True)
    # env.seed(seed)



    ob_shape = list(env.observation_space.shape)
    LOGGER.debug(ob_shape)

    ##create network
    tf.reset_default_graph()
    sess = tf.Session()
    Agent = agent(env, sess, LOGGER, **network_args)
    if Agent.loadModel:
        env = Agent.env
    ## create tensorboard file
    writer = tf.summary.FileWriter(os.path.join(resultsPath,"tensorboard",str(seed)), sess.graph)
    writer.close()

    ##reset enviroment
    obs = env.reset()
    # obs = obs.reshape([1]+ob_shape)


    ##create list for saving
    Rewards = []
    EpisodeRewards = []
    Actions = []
    Observations = []
    Values = []
    allEpR = []
    LogProb = []
    Dones = []
    Timesteps = []
    ElapsedTime = []
    EpisodeLength = []
    pLoss, vLoss=0,0
    ## main loop
    tStart = time.time()
    for timestep in range(numSteps):
        Observations.append(obs)
        #input observation and get the action, logarthmic probabilty and value from the agent
        action, logProb, value, _ = Agent.step(obs)
        # store in lists
        Actions.append(action)
        Values.append(value)
        LogProb.append(logProb)
        # apply action to environment and retreive next observation
        obs, reward, done, info = env.step(action)
        # obs= obs.reshape([1]+ob_shape)

        #store in lists
        Dones.append(done)
        Rewards.append(info['NReward'] if NORMALIZED else reward)
        EpisodeRewards.append(reward)
        # LOGGER.debug("reward: ", nReward, "action: ", action)

        if (timestep+1) % nsteps == 0:

            lr = learningRate(1-(timestep+1-nsteps)/numSteps) # calc current learning rate
            epsilon = clippingFactor(1-(timestep+1-nsteps)/numSteps) #calc current epsilon
            Dones, Rewards, Observations, Actions, Values, LogProb = np.asarray(Dones), np.asarray(Rewards,dtype=np.float32),  np.asarray(Observations,dtype=np.float32).reshape([nsteps]+ob_shape),  np.asarray(Actions,dtype=np.float32),  np.asarray(Values,dtype=np.float32),  np.asarray(LogProb,dtype=np.float32)
            value = Agent.getValue(obs) # get value from latest observation
            # LOGGER.debug(Actions)
            Advantage, DiscRewards = advantageEST(Rewards, Values, Dones, value, gamma,Lamda) #calculate advantange and discounted rewards according to the method in the article
            pLoss, vLoss = Agent.trainNetwork(Observations, Actions, DiscRewards, Values, LogProb, Advantage, lr, epsilon) # train network
            Rewards, Actions, Observations, Values, LogProb, Dones = [],[],[],[],[],[] # create new lists for next batch

        if done: # current episode is finished and needs reset. Also used as a checkpoint for saving intermediate results
            tnow = time.time()
            obs = env.reset()
            # LOGGER.debug("DONE!!: ", EpisodeRewards, sum(EpisodeRewards), len(EpisodeRewards) )
            # obs= obs.reshape([1]+ob_shape)
            latestReward = (sum(EpisodeRewards))
            latestLength = len(EpisodeRewards)
            EpisodeRewards = []
            allEpR.append(latestReward)
            Timesteps.append(timestep)
            ElapsedTime.append(tnow-tStart)
            EpisodeLength.append(latestLength)

        if (timestep+1) % saveInterval == 0: # save current network parameters to disk
            savePath = os.path.join(resultsPath,"checkpoints"+str(timestep+1))
            Agent.saveNetwork(savePath, env)

        if (timestep+1) % logInterval == 0: # print summary to screen
            esttime = (time.time()-tStart)/float(timestep)*(numSteps-timestep)
            esttime = time.strftime("%H:%M:%S", (time.gmtime(esttime)))
            LOGGER.info("average reward: %s", int(np.mean(allEpR[-50:]) if np.isnan(np.mean(allEpR[-50:]))==False else 0))
            LOGGER.info("Latest lenght: %s", int(np.mean(EpisodeLength[-50:]) if np.isnan(np.mean(EpisodeLength[-50:]))==False else 0))
            LOGGER.info("Total episodes: %s", int(len(allEpR)))
            LOGGER.info("Estimated time remaining: %s", esttime)
            LOGGER.info("Time elapsed: %s", time.strftime("%H:%M:%S", (time.gmtime(time.time()-tStart))))
            LOGGER.info("Timestep: %s", timestep+1)
            LOGGER.info("Update %s of %s" ,(timestep+1)/logInterval, numSteps/logInterval)
            LOGGER.info("PolicyLoss: %s \n ValueLoss: %s \n EntropyLoss: %s \n", pLoss, vLoss, 0)

    ttime = time.time()-tStart
    LOGGER.debug("fps: %s", numSteps/(ttime))
    if not loadPath:
        Agent.saveNetwork(os.path.join(resultsPath,"finalModel","final"), env)

        while not done:
            # LOGGER.debug(done)
            obs, reward, done, info = env.step(env.action_space.sample())

    if play and not monitor: # provide short visual render of resulting network
        av = 0
        for _ in range(10):
            rewards = 0
            obs = env.reset()
            done = False
            while not done:
                env.render()
                # time.sleep(1/60.)
                # obs= obs.reshape([1]+ob_shape)
                action, logProb, value, _ = Agent.step(obs)
                obs, reward, done, info = env.step(action)
                rewards +=  reward
                av += reward
            # LOGGER.info("reward: %s", rewards)
        LOGGER.info("averge reward:  %s", av/10.)
    sess.close()
    env.close()

    return allEpR, Timesteps, ElapsedTime, resultsPath

if __name__ == "__main__":


    # for i in range(50000, 350000, 50000):
        # print("checkpoint ", i)
    allEpR, Timesteps, ElapsedTime, resultspath = main(loglevel=logging.INFO)#, loadPath="results/Hopper-v264_2_programtest", checkpoint='checkpoints'+str(i))#)



    if len(allEpR)!=0:

        with open(os.path.join(resultspath,'results.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([allEpR, Timesteps, ElapsedTime], f)


        plt.figure()
        plt.plot(np.arange(len(allEpR)),allEpR)
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.figure()
        plt.plot(Timesteps,allEpR)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')

        plt.figure()
        plt.plot(ElapsedTime,allEpR)
        plt.xlabel('Time [s]')
        plt.ylabel('Reward')

        plt.show()