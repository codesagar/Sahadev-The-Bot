# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
# from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, Basic_Additive_Attn
from modules import  *

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        _, _, num_chars = self.create_char_dicts()
        self.char_vocab = num_chars

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        ## For Char CNN
        self.char_ids_context = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_max_len])
        self.char_ids_qn = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.word_max_len])


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)

    def add_char_embeddings(self):
        """
        Adds char embedding layer to the graph.

        """

        def conv1d(input_, output_size, width, stride, scope_name):
            '''
            :param input_: A tensor of embedded tokens with shape [batch_size,max_length,embedding_size]
            :param output_size: The number of feature maps we'd like to calculate
            :param width: The filter width
            :param stride: The stride
            :return: A tensor of the concolved input with shape [batch_size,max_length,output_size]
            '''
            inputSize = input_.get_shape()[
                -1]  # How many channels on the input (The size of our embedding for instance)

            # This is the kicker where we make our text an image of height 1
            input_ = tf.expand_dims(input_, axis=1)  # Change the shape to [batch_size,1,max_length,output_size]

            # Make sure the height of the filter is 1
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                filter_ = tf.get_variable("conv_filter", shape=[1, width, inputSize, output_size])

            # Run the convolution as if this were an image
            convolved = tf.nn.conv2d(input_, filter=filter_, strides=[1, 1, stride, 1], padding="VALID")
            # Remove the extra dimension, eg make the shape [batch_size,max_length,output_size]
            result = tf.squeeze(convolved, axis=1)
            return result

        with vs.variable_scope("char_embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            char_emb_matrix = tf.Variable(tf.random_uniform((self.char_vocab, self.FLAGS.char_embedding_size), -1, 1)) #is trainable

            print("Shape context placeholder", self.char_ids_context.shape)
            print("Shape qn placeholder", self.char_ids_qn.shape)

            self.context_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, tf.reshape(self.char_ids_context, shape=(-1, self.FLAGS.word_max_len))) # shape (-1, word_max_len, char_embedding_size)

            ##reshape to 3d tensor - compress dimensions we don't want to convolve on
            self.context_char_embs = tf.reshape(self.context_char_embs, shape=(
           -1, self.FLAGS.word_max_len, self.FLAGS.char_embedding_size))  # shape = batch_size*context_len, word_max_len, char_embedding_size

            print("Shape context embs before conv", self.context_char_embs.shape)

            ## Repeat for question embeddings - again reshape to 3D tensor
            self.qn_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, tf.reshape(self.char_ids_qn, shape=(-1, self.FLAGS.word_max_len)))
            self.qn_char_embs = tf.reshape(self.qn_char_embs, shape=(-1, self.FLAGS.word_max_len, self.FLAGS.char_embedding_size))


            print("Shape qn embs before conv", self.qn_char_embs.shape)

            ## Now implement convolution. I decided to use conv2d through the function conv1d above since that was more intuitive

            self.context_emb_out = conv1d(input_= self.context_char_embs, output_size = self.FLAGS.char_out_size,  width =self.FLAGS.window_width , stride=1, scope_name='char-cnn')

            self.context_emb_out = tf.nn.dropout(self.context_emb_out, self.keep_prob)

            print("Shape context embs after conv", self.context_emb_out.shape)

            self.context_emb_out = tf.reduce_sum(self.context_emb_out, axis = 1)

            self.context_emb_out =  tf.reshape(self.context_emb_out, shape=(-1, self.FLAGS.context_len, self.FLAGS.char_out_size))# Desired shape is Batch_size, context_len, char_out_size

            print("Shape context embs after pooling", self.context_emb_out.shape)

            self.qn_emb_out = conv1d(input_=self.qn_char_embs, output_size=self.FLAGS.char_out_size,
                                          width=self.FLAGS.window_width, stride=1, scope_name='char-cnn') #reuse weights b/w context and ques conv layers

            self.qn_emb_out = tf.nn.dropout(self.qn_emb_out, self.keep_prob)

            print("Shape qn embs after conv", self.qn_emb_out.shape)

            self.qn_emb_out = tf.reduce_sum(self.qn_emb_out,
                                                 axis=1)

            self.qn_emb_out = tf.reshape(self.qn_emb_out, shape=(-1, self.FLAGS.question_len,
                                                                           self.FLAGS.char_out_size))  # Desired shape is Batch_size, question_len, char_out_size

            print("Shape qn embs after pooling", self.qn_emb_out.shape)

            return self.context_emb_out, self.qn_emb_out

    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.


        ###################################################### CHAR EMBEDDING   #######################################
        if self.FLAGS.do_char_embed:

            self.context_emb_out, self.qn_emb_out = self.add_char_embeddings()
            self.context_embs = tf.concat((self.context_embs, self.context_emb_out), axis = 2)
            print("Shape - concatenated context embs", self.context_embs.shape)

            self.qn_embs = tf.concat((self.qn_embs, self.qn_emb_out), axis=2)
            print("Shape - concatenated qn embs", self.qn_embs.shape)

        ###################################################### HIGHWAY LAYER   #######################################
        if self.FLAGS.add_highway_layer:
            last_dim_concat = self.context_embs.get_shape().as_list()[-1]
            for i in range(2):
                #add two highway layers or repeat process twice
                self.context_embs = self.highway(self.context_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)
                #reuse variables for qn_embs
                self.qn_embs = self.highway(self.qn_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)

        ###################################################### RNN ENCODER  #######################################
        encoder = RNNEncoder(self.FLAGS.hidden_size_encoder, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask, scopename='RNNEncoder') # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask, scopename='RNNEncoder') # (batch_size, question_len, hidden_size*2)

        ###################################################### CNN ENCODER  #######################################
        if self.FLAGS.cnn_encoder:
            ## Use CNN to also generate encodings
            cnn_encoder = CNNEncoder(self.FLAGS.filter_size_encoder, self.keep_prob)
            context_cnn_hiddens = cnn_encoder.build_graph(self.context_embs, self.FLAGS.context_len, scope_name='context-encoder')  # (batch_size, context_len, hidden_size*2)
            print("Shape - Context Encoder output", context_cnn_hiddens.shape)

            ques_cnn_hiddens = cnn_encoder.build_graph(self.qn_embs, self.FLAGS.question_len,
                                                       scope_name='ques-encoder')  # (batch_size, context_len, hidden_size*2)
            print("Shape - Ques Encoder output", ques_cnn_hiddens.shape)

            ## concat these vectors

            context_hiddens = context_cnn_hiddens
            question_hiddens = ques_cnn_hiddens
            # context_hiddens = tf.concat((context_hiddens, context_cnn_hiddens), axis = 2)  # Just use these output for now. Ignore the RNN output
            # question_hiddens = tf.concat((question_hiddens, ques_cnn_hiddens), axis = 2)   # Just use these output for now
            print("Shape - Context Hiddens", context_hiddens.shape)

        ###################################################### RNET QUESTION CONTEXT ATTENTION and SELF ATTENTION  #######################################
        if self.FLAGS.rnet_attention:  ##perform Question Passage and Self Matching attention from R-Net

            rnet_layer = Attention_Match_RNN(self.keep_prob, self.FLAGS.hidden_size_encoder, self.FLAGS.hidden_size_qp_matching, self.FLAGS.hidden_size_sm_matching)

           # Implement better question_passage matching
            v_P = rnet_layer.build_graph_qp_matching(context_hiddens, question_hiddens, self.qn_mask, self.context_mask, self.FLAGS.context_len, self.FLAGS.question_len)

            self.rnet_attention = v_P

            self.rnet_attention = tf.squeeze(self.rnet_attention, axis=[2])  # shape (batch_size, seq_len)

            # Take softmax over sequence
            _, self.rnet_attention_probs = masked_softmax(self.rnet_attention, self.context_mask, 1)

            h_P = rnet_layer.build_graph_sm_matching(context_hiddens, question_hiddens, self.qn_mask, self.context_mask,
                                                     self.FLAGS.context_len, self.FLAGS.question_len, v_P)

            # Blended reps for R-Net
            blended_reps = tf.concat([context_hiddens, v_P, h_P], axis=2)  # (batch_size, context_len, hidden_size*6)

        ###################################################### BIDAF ATTENTION AND MODELING LAYER  #######################################
        elif self.FLAGS.bidaf_attention:

            attn_layer = BiDAF(self.keep_prob, self.FLAGS.hidden_size_encoder * 2)
            attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens,
                                                 self.context_mask)  # attn_output is shape (batch_size, context_len, hidden_size_encoder*6)

            self.bidaf_attention = attn_output
            self.bidaf_attention = tf.reduce_max(self.bidaf_attention, axis=2)  # shape (batch_size, seq_len)
            print("Shape bidaf before softmax", self.bidaf_attention.shape)

            # Take softmax over sequence
            _, self.bidaf_attention_probs = masked_softmax(self.bidaf_attention, self.context_mask, 1)  ## for plotting purpose


            blended_reps = tf.concat([context_hiddens, attn_output], axis=2)  # (batch_size, context_len, hidden_size_encoder*8)

            ## add a modeling layer
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
            attention_hidden = modeling_layer.build_graph(blended_reps,
                                                  self.context_mask, scopename='bidaf_modeling')  # (batch_size, context_len, hidden_size*2)

            blended_reps = attention_hidden # for the final layer

        ###################################################### BASELINE DOT PRODUCT ATTENTION  #######################################
        else: ## perform baseline dot product attention

            # Use context hidden states to attend to question hidden states - Basic Attention

            last_dim = context_hiddens.get_shape().as_list()[-1]
            print("last dim", last_dim)

            attn_layer = BasicAttn(self.keep_prob, last_dim,
                                   last_dim)
            _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask,
                                                    context_hiddens)  # attn_output is shape (batch_size, context_len, hidden_size*2)


            # Concat attn_output to context_hiddens to get blended_reps
            blended_reps = tf.concat([context_hiddens, attn_output], axis=2)  # (batch_size, context_len, hidden_size*4)

        ###################################################### RNET QUESTION POOLING and ANSWER POINTER  #######################################
        if self.FLAGS.answer_pointer_RNET:  ##Use Answer Pointer Module from R-Net

            if self.FLAGS.rnet_attention:
                # different attention size for R-Net final layer
                hidden_size_attn = 2 * self.FLAGS.hidden_size_encoder + self.FLAGS.hidden_size_qp_matching + 2 * self.FLAGS.hidden_size_sm_matching  # combined size of blended reps

            elif self.FLAGS.bidaf_attention:
                hidden_size_attn = 2*self.FLAGS.hidden_size_modeling

            else:
                hidden_size_attn = 4 * self.FLAGS.hidden_size_encoder # Final attention size for baseline model

            attn_ptr_layer = Answer_Pointer(self.keep_prob, self.FLAGS.hidden_size_encoder,
                                            self.FLAGS.question_len, hidden_size_attn)

            p, logits = attn_ptr_layer.build_graph_answer_pointer(context_hiddens, question_hiddens, self.qn_mask,
                                                                  self.context_mask,
                                                                  self.FLAGS.context_len, self.FLAGS.question_len,
                                                                  blended_reps)

            self.logits_start = logits[0]
            self.probdist_start = p[0]

            self.logits_end = logits[1]
            self.probdist_end = p[1]

        ###################################################### BASELINE FULLY CONNECTED ANSWER PREDICTION  #######################################
        else: # Use baseline fully connected for answer start and end prediction

            # Apply fully connected layer to each blended representation
            # Note, blended_reps_final corresponds to b' in the handout
            # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
            blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size_fully_connected) # blended_reps_final is shape (batch_size, context_len, hidden_size)

            # Use softmax layer to compute probability distribution for start location
            # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
            with vs.variable_scope("StartDist"):
                softmax_layer_start = SimpleSoftmaxLayer()
                self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

            # Use softmax layer to compute probability distribution for end location
            # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
            with vs.variable_scope("EndDist"):
                softmax_layer_end = SimpleSoftmaxLayer()
                self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout
        if self.FLAGS.do_char_embed:
            input_feed[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
            input_feed[self.char_ids_qn] = self.padded_char_ids(batch, batch.qn_ids)

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        if self.FLAGS.do_char_embed:
            input_feed[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
            input_feed[self.char_ids_qn] = self.padded_char_ids(batch, batch.qn_ids)

        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        if self.FLAGS.do_char_embed:
            input_feed[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
            input_feed[self.char_ids_qn] = self.padded_char_ids(batch, batch.qn_ids)

        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)
        maxprob = 0
        ###################################################### SMARTER SPAN SELECTION  #######################################
        if self.FLAGS.smart_span:

            curr_batch_size = batch.batch_size
            start_pos = np.empty(shape = (curr_batch_size), dtype=int)
            end_pos = np.empty(shape=(curr_batch_size), dtype=int)
            maxprob = np.empty(shape=(curr_batch_size), dtype=float)

            for j in range(curr_batch_size):  # for each row
            ## Take argmax of start and end dist in a window such that  i <= j <= i + 15
                maxprod = 0
                chosen_start = 0
                chosen_end = 0
                for i in range(self.FLAGS.context_len-16):
                    end_dist_subset = end_dist[j,i:i+16]
                    end_prob_max = np.amax(end_dist_subset)
                    end_idx = np.argmax(end_dist_subset)
                    start_prob = start_dist[j,i]
                    prod = end_prob_max*start_prob
                    # print("Prod: ", prod)

                    # print("Shape end, start:", end_prob_max.shape, start_prob.shape)

                    if prod > maxprod:
                        maxprod = prod
                        chosen_start = i
                        chosen_end = chosen_start+end_idx

                start_pos[j] = chosen_start
                # end_idx = np.argmax(end_dist[j:chosen_start:chosen_start+16])
                # print("Chosen end", chosen_start+end_idx)
                end_pos[j] = chosen_end
                maxprob[j] = round(maxprod,4)

                ## add sanity check
                delta = end_pos[j] - start_pos[j]
                if delta < 0 or delta > 16:
                    print("Error! Please look ")

                # print("Shape end, start matrix:", start_pos.shape, end_pos.shape)
                # print("Start and end matrix:", start_pos,  end_pos)
                # print("Maxprob: ", maxprob.shape, maxprob)

        else:

            # Take argmax to get start_pos and end_post, both shape (batch_size)
            start_pos = np.argmax(start_dist, axis=1)
            end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos, maxprob

    def get_attention_dist(self, session, batch):

        # input_feed = {}
        # input_feed[self.context_ids] = batch.context_ids
        # input_feed[self.context_mask] = batch.context_mask
        # input_feed[self.qn_ids] = batch.qn_ids
        # input_feed[self.qn_mask] = batch.qn_mask
        # if self.FLAGS.do_char_embed:
        #     input_feed[self.char_ids_context] = self.padded_char_ids(batch, batch.context_ids)
        #     input_feed[self.char_ids_qn] = self.padded_char_ids(batch, batch.qn_ids)
        #
        #
        # if self.FLAGS.rnet_attention:
        #     output_feed = [self.rnet_attention_probs]
        #
        # elif self.FLAGS.bidaf_attention:
        #     output_feed = [self.bidaf_attention_probs]
        #
        # [attn_distribution] = session.run(output_feed, input_feed)

        start_dist, end_dist = self.get_prob_dists(session, batch)

        return start_dist

    ## Helper Functions for Char CNN

    def create_char_dicts(self, CHAR_PAD_ID=0, CHAR_UNK_ID = 1, _CHAR_PAD = '*', _CHAR_UNK = '$' ):

        unique_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                        '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '[', ']', '^', 'a', 'b', 'c', 'd',
                        'e' , 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        '~', ]  # based on analysis in jupyter notebook

        num_chars = len(unique_chars)

        idx2char = dict(enumerate(unique_chars, 2))  ##reserve first 2 spots
        idx2char[CHAR_PAD_ID] = _CHAR_PAD
        idx2char[CHAR_UNK_ID] = _CHAR_UNK

        ##Create reverse char2idx
        char2idx = {v: k for k, v in idx2char.iteritems()}
        return char2idx, idx2char, num_chars

    def word_to_token_ids(self, word):
        """Turns a word into char idxs
            e.g. "know" -> [9, 32, 16, 96]
            Note any token that isn't in the char2idx mapping gets mapped to the id for UNK_CHAR
            """
        char2idx, idx2char, _ = self.create_char_dicts()
        char_tokens = list(word)  # list of chars in word
        char_ids = [char2idx.get(w, 1) for w in char_tokens]
        return char_tokens, char_ids


    def padded_char_ids(self,batch, token_ids):  # have to use token_ids since only those are padded

        charids_batch = []
        for i in range(batch.batch_size):
            charids_line = []
            #for each example
            token_row = token_ids[i,:]
            # print("Each token row is", token_row)
            # print("Shape token row is ", token_row.shape)
            for j in range(len(token_row)):
                id = token_row[j]
                # print("each id is:" ,id)
                word = self.id2word[id] # convert token id to word
                _, char_ids = self.word_to_token_ids(word)
                # for each word we get char_ids but they maybe different_length
                if len(char_ids) < self.FLAGS.word_max_len: #pad with CHAR pad tokens
                    while len(char_ids) < self.FLAGS.word_max_len:
                        char_ids.append(0)
                    pad_char_ids = char_ids

                else:  # if longer truncate to word max len
                    pad_char_ids = char_ids[:self.FLAGS.word_max_len]

                charids_line.append(pad_char_ids)
            charids_batch.append(charids_line)

        return charids_batch

    def matrix_multiplication(self, mat, weight):
        # [batch_size, seq_len, hidden_size] * [hidden_size, p] = [batch_size, seq_len, p]

        mat_shape = mat.get_shape().as_list()  # shape - ijk
        weight_shape = weight.get_shape().as_list()  # shape -kl
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])  # reshape to batch_size, seq_len, p

    def highway(self, x, size, scope_name, carry_bias=-1.0):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
            b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

            W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
            b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")

        T = tf.sigmoid(self.matrix_multiplication(x, W_T) + b_T, name="transform_gate")
        H = tf.nn.relu(self.matrix_multiplication(x, W) + b, name="activation")

        print("shape H, T: ", H.shape, T.shape)
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, 
            self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        if total_num_examples==0:
            print(total_num_examples)
            total_num_examples += 1
            dev_loss =  0.001
        else:
            dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos, _ = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= (example_num+0.0001)
        em_total /= (example_num+0.0001)

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
