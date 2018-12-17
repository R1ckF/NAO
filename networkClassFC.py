import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import gym
from gym import spaces
import pickle

class network:
    """
    This class is used as a framework that can be called to create actor / critic networks.
    env: gym environment (needed to obtain outputspace of network to match input of env)
    networkOption: the network that is selected (only fc is used as cnn's took to long to trainNetwork
    sess: tensorflow session used to run the network
    """
    def __init__(self,env, networkOption, sess, LOGGER):
        self.env = env
        self.networkOption = networkOption
        self.LOGGER=LOGGER
        if isinstance(self.env.action_space, spaces.Discrete):
            self.outputShape = (self.env.action_space.n)
            self.logP = self.logPDiscrete


        elif isinstance(self.env.action_space, spaces.Box):
            self.outputShape = (self.env.action_space.shape[0])
            self.logP = self.logPContinuous
        else: self.LOGGER.warning("ERROR!!!!! outputshape undetermined")
        ## function that creates a convolutional layer. Just reorganised it a bit for easier typing
    def conv(self, input, outputDepth, kernelsize, stride, padding="valid", name = 'Conv', **network_args):
        return tf.layers.conv2d(
          inputs=input,
          filters=outputDepth,
          kernel_size=[kernelsize, kernelsize],strides=(stride,stride),padding=padding, name = name, **network_args)

          ##Function that creates a fully connected layer. Just reorganised it a bit for easier typing
    def fc(self, input, numOutputs, name = None, **network_args):
        return tf.layers.dense(
            input,
            numOutputs,
            name = name, **network_args)

        ## building cnnsmall consisting of 2 convolutional layers and 1 FC layer
    def cnnSmall(self, observationPH, **network_args):
        self.LOGGER.debug("obsPH: %s",observationPH.shape)
        outputC1 = self.conv(observationPH, 8, 8, 4, name = 'Conv1', **network_args)
        self.LOGGER.debug("CNN1: %s",outputC1.shape)
        outputC2 = self.conv(outputC1, 16, 4, 2, name = 'Conv2', **network_args)
        self.LOGGER.debug("CNN2: %s",outputC2.shape)
        outputFlatten = tf.layers.flatten(outputC2)
        self.LOGGER.debug("Flatten: %s",outputFlatten.shape)
        outputFC = self.fc(outputFlatten, 128, name= 'FC1', **network_args)
        self.LOGGER.debug("networkOutput: %s",outputFC.shape)
        return outputFC

        ## building cnnlarge consisting of 3 deeper convolutional layers and 1 large FC layer
    def cnnLarge(self, observationPH, **network_args):
        self.LOGGER.debug("obsPH: %s",observationPH.shape)
        outputC1 = self.conv(observationPH, 32, 8, 4, name = 'Conv1', **network_args)
        self.LOGGER.debug("CNN1: %s",outputC1.shape)
        outputC2 = self.conv(outputC1, 64, 4, 2, name = 'Conv2', **network_args)
        self.LOGGER.debug("CNN2: %s",outputC2.shape)
        outputC3 = self.conv(outputC1, 64, 3, 1, name = 'Conv3', **network_args)
        self.LOGGER.debug("CNN3: %s",outputC3.shape)
        outputFlatten = tf.layers.flatten(outputC3)
        self.LOGGER.debug("Flatten: %s", outputFlatten.shape)
        outputFC = self.fc(outputFlatten, 512, name= 'FC', **network_args)
        self.LOGGER.debug("networkOutput: %s",outputFC.shape)
        return outputFC

        ##building Fully connected network with amount of layers and nodes as given by network_args.
        ## 2 layers and 64 units corresponds to the test that were completed in the PPO paper
    def fcNetwork(self, observationPH, numNodes = [64,64], **network_args):
        self.LOGGER.debug("obsPH: %s",observationPH.shape)
        vector = observationPH
        for i,nodes in enumerate(numNodes):
            vector = self.fc(vector, nodes, **network_args)
            self.LOGGER.debug("layer"+str(i)+": %s",vector.shape)
        return vector

    def buildNetwork(self,observationPH,**network_args):  ## select which network to build
        if self.networkOption == 'small':
            self.LOGGER.debug('Small network selected')
            self.networkOutput=self.cnnSmall(observationPH,**network_args)
        elif self.networkOption == 'large':
            self.LOGGER.debug('Large network selected')
            self.networkOutput=self.cnnLarge(observationPH,**network_args)
        elif self.networkOption == 'fc':
            self.networkOutput=self.fcNetwork(observationPH, **network_args)
        else:
            raise ValueError('Invalid network option')

    def createStep(self, **network_args):  ## depending on actor or critic mode, provide the appropiate step output
        if tf.get_variable_scope().name=='actor': ## actor determines the action and the probabilty of that action
            if isinstance(self.env.action_space, spaces.Discrete):
                old = network_args['activation']
                network_args['activation'] = None
                self.actionOutput = self.fc(self.networkOutput,self.outputShape, **network_args)
                self.meanActionOutput = tf.argmax(tf.squeeze(self.actionOutput))
                self.LOGGER.debug("actionspace output: %s", self.actionOutput.shape)
                self.action = tf.multinomial(self.actionOutput,1)
                self.LOGGER.debug("action shape: %s", self.action.shape)
                self.logProb = self.logP(self.action)
                self.LOGGER.debug("logprob shape: %s", self.logProb.shape)
                network_args['activation'] = old

            elif  isinstance(self.env.action_space, spaces.Box):
                old = network_args['activation']
                network_args['activation'] = None
                self.meanActionOutput = self.fc(self.networkOutput,self.outputShape, **network_args)
                self.logstd = tf.get_variable(name='logstd', shape=[1, self.outputShape], initializer=tf.zeros_initializer())
                self.std = tf.exp(self.logstd)
                self.action = self.meanActionOutput + self.std * tf.random_normal(tf.shape(self.meanActionOutput))
                # self.action = tf.Print(self.action, [self.meanActionOutput,self.std, tf.random_normal(tf.shape(self.meanActionOutput),seed=0) ])
                self.LOGGER.debug("action shape: %s", self.action.shape)
                self.logProb = self.logP(self.action)
                self.LOGGER.debug("logprob shape: %s", self.logProb.shape)

                network_args['activation'] = old

        elif tf.get_variable_scope().name=='critic': ## critic determines the value of the current state
            old = network_args['activation']
            network_args['activation'] = None
            self.value = self.fc(self.networkOutput, 1, name='Value', **network_args)
            self.LOGGER.debug("Value shape: %s", self.value.shape)
            network_args['activation'] = old
        else:
            raise ValueError('no scope detected')


    def logPDiscrete(self, action): ## function needed to calculate the probabilty of a selected action, both used during stepping trough the environment and during training.

        one_hot_actions = tf.one_hot(action,self.outputShape)
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actionOutput,labels=one_hot_actions)


    def logPContinuous(self, action):
        return 0.5 * tf.reduce_sum(tf.square((action - self.meanActionOutput) / self.std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(action)[-1]) \
            + tf.reduce_sum(self.logstd, axis=-1)
class agent:
    """
    Provides an agent model that is used to create networks and determining actions based on the observations
    Also provides training functionality for the networks
    env: gym environments
    sess: tensorflow Session
    networkoption: which type of network to created, for this research only fc is used
    epochs: number of training epochs
    nMinibatch: amount of minibatches that are created from each training set
    loadModel: can be used to load a previously created tensorflow model
    networkStyle: 'copy' uses 2 different networks, 'shared' uses the same network but only uses different output layers for action and value network
    c1: weight of the value function in the total loss function

    """

    def __init__(self, env, sess, LOGGER, networkOption='fc', epochs = 4, nMiniBatch = 2, loadModel = None,
                    networkStyle = 'copy', c1=0.5, **network_args):
        self.env = env
        self.sess = sess
        self.networkOption = networkOption
        self.epoch = epochs
        self.nMiniBatch = nMiniBatch
        self.loadModel  = loadModel
        self.LOGGER = LOGGER
        self.shp = list(self.env.observation_space.shape)
        self.actionShp= [1] if list(self.env.action_space.shape)==() else list(self.env.action_space.shape)
        ## Create placeholders used to feed data
        self.observationPH = tf.placeholder(tf.float32,shape=[None]+self.shp, name = "Observation")#,self.shp[1],self.shp[2]]
        if isinstance(env.action_space, spaces.Discrete): self.actionsPH = tf.placeholder(tf.int32,shape=[None]+self.actionShp,name='Actions')
        elif isinstance(env.action_space, spaces.Box): self.actionsPH = tf.placeholder(tf.float32,shape=[None]+self.actionShp,name='Actions')
        self.LOGGER.debug(self.actionsPH.dtype)
        self.actionsProbOldPH = tf.placeholder(tf.float32,shape=[None],name='ActionProbOld')
        self.advantagePH = tf.placeholder(tf.float32, shape=[None],name='Advantage')
        self.disRewardsPH = tf.placeholder(tf.float32, shape = [None,1], name = 'DiscountedRewards')
        self.oldValuePredPH = tf.placeholder(tf.float32, shape = [None,1], name = 'oldValuePred')
        self.learningRatePH = tf.placeholder(tf.float32, shape = [], name = 'LearningRate')
        self.epsilonPH = tf.placeholder(tf.float32, shape = [], name = 'ClippingFactor')
        if networkStyle == 'copy': ## build netwok using the network class, the scopes what network is created
            with tf.variable_scope('actor'):
                network_args['trainable']=True
                old = network_args.copy()
                self.LOGGER.debug(network_args)
                self.actor = network(self.env, self.networkOption, self.sess, self.LOGGER)
                self.actor.buildNetwork(self.observationPH,**network_args)
                del network_args['numNodes']
                # del network_args['numLayers']
                self.actor.createStep(**network_args)
                self.meanActionOutput = self.actor.meanActionOutput
                self.action = self.actor.action
                self.logProb = self.actor.logProb
                self.logP = self.actor.logP
                self.LOGGER.debug(network_args)
                network_args = old
                self.LOGGER.debug(network_args)

            with tf.variable_scope('critic'):
                network_args['trainable']=True
                self.LOGGER.debug(network_args)
                self.critic = network(self.env, self.networkOption, self.sess, self.LOGGER)
                self.critic.buildNetwork(self.observationPH,**network_args)
                del network_args['numNodes']
                # del network_args['numLayers']
                self.critic.createStep(**network_args)
                self.value = self.critic.value

        elif networkStyle == 'shared':
            with tf.variable_scope('actor'):
                network_args['trainable']=True
                self.shared = network(self.env, self.networkOption, self.sess, self.LOGGER)
                self.shared.buildNetwork(self.observationPH,**network_args)
                self.shared.createStep(**network_args)
                self.action = self.shared.action
                self.meanActionOutput = self.shared.meanActionOutput
                self.logProb = self.shared.logProb
                self.logP = self.shared.logP

            with tf.variable_scope('critic'):
                network_args['trainable']=True
                self.shared.createStep(**network_args)
                self.value = self.shared.value

        else:
            raise ValueError('networkStyle not recognized')

        with tf.variable_scope('lossFunction'): ## create loss function according to article
            actionsProbNew = self.logP(self.actionsPH) #determine prob of each action given the current network. The old probabilities were already found during the stepping phase
            self.LOGGER.debug("NewProb: %s", actionsProbNew.shape)
            ratio = tf.exp(self.actionsProbOldPH - actionsProbNew)  # determine ratio (newProb/oldProb)#negative logprop is given by tensorflow thats why they are switched.. Took me a while to find this out
            self.LOGGER.debug("ratio: %s", ratio.shape)
            policyLoss= ratio * self.advantagePH # calculate unclipped policy loss
            self.LOGGER.debug("Policylos: %s", policyLoss.shape)
            clippedPolicyLoss= self.advantagePH * tf.clip_by_value(ratio,(1-self.epsilonPH),(1+self.epsilonPH)) # calculate clipped policy loss
            self.LOGGER.debug("clippedPolicyLoss: %s", clippedPolicyLoss.shape)
            min = tf.minimum(policyLoss, clippedPolicyLoss) # find the minimum of the clipped and unclipped loss
            self.LOGGER.debug("minShape: %s", min.shape)
            self.pLoss = -tf.reduce_mean(min) # calculate the average of the entire batch
            self.LOGGER.debug("pLoss: %s", self.pLoss.shape)

            value = self.value # calculate value using current network. Old values are already determined in the stepping phase
            self.LOGGER.debug("value: %s", self.value.shape)
            valueLoss = tf.square(value - self.disRewardsPH) # calculate squared error value loss
            self.LOGGER.debug("valueLoss: %s", valueLoss.shape)
            clippedValue = self.oldValuePredPH + tf.clip_by_value(self.value - self.oldValuePredPH, -self.epsilonPH, self.epsilonPH)
            clippedValueLoss = tf.square(clippedValue - self.disRewardsPH)
            self.vLoss = 0.5 * tf.reduce_mean(tf.maximum(valueLoss, clippedValueLoss)) # calculate average valueloss over entire batch
            self.LOGGER.debug("vLoss: %s", self.vLoss.shape)
            # self.pLoss = tf.Print(self.pLoss,[tf.maximum(valueLoss, clippedValueLoss), self.vLoss], summarize=1000)
            #  [tf.shape(tf.maximum(valueLoss, clippedValueLoss)),\
            #  tf.shape(value), tf.shape(self.oldValuePredPH), tf.shape(self.actionsProbOldPH), tf.shape(actionsProbNew), \
            #  tf.shape(self.actionsPH), tf.shape(ratio), tf.shape(self.advantagePH)], summarize=1000)
            self.loss = self.pLoss + c1 * self.vLoss # total loss function

            self.LOGGER.debug("loss: %s",self.loss.shape)

        with tf.variable_scope('trainer'): ## create trainer using adam and a learning rate as given
            self.train = tf.train.AdamOptimizer(learning_rate= self.learningRatePH).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=100) ## create saver that allows to save and restore the model
        self.LOGGER.debug('Agent created with following properties: %s, %s', self.__dict__, network_args)

        if loadModel: ## load model if load path is given
            self.saver.restore(self.sess, self.loadModel+'.ckpt')
            self.LOGGER.info("Model loaded from %s", self.loadModel)
            with open(self.loadModel+'.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
                self.env = pickle.load(f)
        else:
            self.sess.run(tf.global_variables_initializer())

    def step(self, observation):
        # function that uses the observation to calculate the action, logProbabilty and value of the network and returns them
        action, logProb, value, meanAction = self.sess.run([self.action,self.logProb, self.value, self.meanActionOutput], feed_dict= {self.observationPH : observation})
        return  action.squeeze(), logProb.squeeze(), value.squeeze(), meanAction.squeeze()


    def trainNetwork(self, observations, actions, disRewards, values, actionProbOld, advantage,lr, epsilon):
        """
        Functions that first creates shuffled minibatches of the data.
        Next it trains the network on those batches for a given amount of epochs
        returns the latest policy and value loss
        """
        lenght = observations.shape[0]
        step = int(lenght/self.nMiniBatch)
        assert(self.nMiniBatch*step == lenght)
        indices = range(0,lenght,step)
        randomIndex = np.arange(lenght)
        for _ in range(self.epoch):
            np.random.shuffle(randomIndex)
            for start in indices:
                end = start+step
                ind = randomIndex[start:end].astype(np.int32)
                observationsB = observations[ind]
                actionsB = actions[ind]
                actionProbOldB = actionProbOld[ind]
                advantageB = advantage[ind]
                disRewardsB = disRewards[ind]
                valuesB = values[ind]
                advantageB = (advantageB - advantageB.mean()) / (advantageB.std() + 1e-8) # normalize the advantage function for faster convergence
                feedDict = {self.observationPH: observationsB, self.oldValuePredPH:valuesB.reshape((-1,1)), self.actionsPH: actionsB, self.actionsProbOldPH: actionProbOldB, self.advantagePH: advantageB, self.disRewardsPH: disRewardsB.reshape((-1,1)), self.learningRatePH: lr, self.epsilonPH: epsilon}
                ## feedDict is used to feed the training data to the placeholders created when the agent is initialized
                # self.LOGGER.debug(feedDict)
                pLoss, vLoss,  _ = self.sess.run([self.pLoss, self.vLoss, self.train], feedDict)

        return pLoss, vLoss

    def getValue(self, observation): ## functions used to get the value of the current observations. Needed for the advantage calculation
        return (self.sess.run(self.value,{self.observationPH: observation}))

    def saveNetwork(self,name, env): ## function that saves the current network parameters
        savePath = self.saver.save(self.sess,name+'.ckpt')
        with open((name+'.pkl'), 'wb') as f:
            pickle.dump(env.envL[0], f)
        self.LOGGER.info("Model saved in path: %s" % savePath)



## other definitions

def advantageEST(rewards, values, dones, lastValue, gamma, lamda):

    ## using advantage estimator from article
    # self.LOGGER.debug("rewards: ", rewards)
    advantage = np.zeros_like(rewards).astype(np.float32)
    advantage[-1] = lastValue*gamma * (1-dones[-1])+rewards[-1]-values[-1] # calculate latest advantage
    lastAdv = advantage[-1]
    for index in reversed(range(len(rewards)-1)):
        ## Doing it in reverse allows for reuse of already calculated variables and is much faster
        delta = rewards[index] + gamma * (1-dones[index])* values[index+1] -values[index]
        # when an environment was done. The last advantage needs to be ignored as a new episoded begins and the rewards are 0 again
        advantage[index] = lastAdv = delta + gamma * (1-dones[index]) * lamda * lastAdv
    return advantage, (advantage+values)  ## advantage+values= discountedReward according to the article
