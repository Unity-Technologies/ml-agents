# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:28:50 2017

@author: tcmxx
"""



import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import os

    
class agent():
    def __init__(self, lr):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        # parameters
        env_s_size = 36
        hidden_size = 256
        action_size=4
        self.input_batchsize= tf.placeholder(shape=[],dtype=tf.int32, name = 'batch_size')
        #first, the grid data layer. using conv
        self.env_state_in= tf.placeholder(shape=[None,env_s_size],dtype=tf.float32, name = 'input_state')
        hidden = self.env_state_in
        #for i in range(h_num):
        #    hidden = slim.fully_connected(hidden,hidden_size,biases_initializer=None,activation_fn=tf.nn.relu)
        #self.env_final = slim.fully_connected(hidden,hidden_size,activation_fn=tf.nn.relu);
        
        convin = tf.reshape(hidden, [-1, 6, 6, 1])
        conv1 = tf.layers.conv2d(inputs=convin,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1,filters=hidden_size,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)
        self.env_final = tf.reshape(pool2, [-1, hidden_size])
        
    
        
        #seperate Advantage and Value parts of the Q network
        self.streamA,self.streamV = tf.split(self.env_final,2,1)        
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([hidden_size//2,action_size]))
        self.VW = tf.Variable(xavier_init([hidden_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.Qout =  tf.identity(self.Qout, name="output_Qs")
        self.Weightout =  tf.nn.softmax(self.Qout, name="output_weight")
        self.chosen_action = tf.expand_dims(tf.argmax(self.Qout,1), name = 'chosen_action', axis= 1)
        self.max_Q = tf.reduce_max(self.Qout,axis = 1, name = 'max_Qs')
        
        #training part
        self.targetQ_holder = tf.placeholder(shape=[None],dtype=tf.float32, name = 'input_targetQ')
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name = 'input_action')
        self.action_onehot = tf.one_hot(self.action_holder,action_size,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_onehot), axis=1)# evaluated Q
        
        self.td_error = tf.square(self.targetQ_holder - self.Q)
        self.loss = tf.reduce_mean(self.td_error, name = 'output_loss')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum = 0.9, use_nesterov=False)
        self.update_once = optimizer.minimize(self.loss,name = 'train_once')
    
    
        #%% training
        init = tf.global_variables_initializer()
        output_dir = 'QNMaze1'
        #save the graph and create the save/restore op
        with tf.Session() as sess:
            sess.run(init)
            #%%###############################################
            #save the check points and print the file/operation name for restoring in C#
            saver = tf.train.Saver()
            saver_def = saver.as_saver_def()
            output_dir =  os.path.join(os.getcwd(),output_dir)
            saver.save(sess, os.path.join(output_dir ,'checkpoints/modeldata'))
            # The name of the tensor you must feed with a filename when saving/restoring.
            print(saver_def.filename_tensor_name)
            # The name of the target operation you must run when restoring.
            print(saver_def.restore_op_name)
            # The name of the target operation you must run when saving.
            print(saver_def.save_tensor_name)
            print(tf.train.latest_checkpoint(os.path.join(output_dir ,'checkpoints')))
            
            #%%######################################
            #write the graph to binary and txt form
            tf.train.write_graph(sess.graph,output_dir ,'ql_maze.bytes',as_text=False);
            tf.train.write_graph(sess.graph,output_dir ,'ql_maze.txt',as_text=False);
            
            
            #%%#######################################3
            #write the summary for tensorboard
            train_writer = tf.summary.FileWriter(os.path.join(output_dir ,'logs'), sess.graph)
            train_writer.close()  
            
            
            
tf.reset_default_graph() #Clear the Tensorflow graph.x

myAgent = agent(lr=0.02) #Load the agent.

