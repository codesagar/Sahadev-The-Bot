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

"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
import io
import re
import json
import sys
import logging
import numpy as np
import flask
import pickle
from flask_cors import CORS
import logging

from random import randint

from flask import Flask, render_template

from flask import request, session

from flask_ask import (
	Ask,
	statement,
	question,
	request as ask_request,
	session as ask_session,
	version
)

import tensorflow as tf

from qa_model import QAModel
from vocab import get_glove
from official_eval_helper import get_json_data, generate_answers, generate_answers_prob


logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "official_eval", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "bidaf_best", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 60, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size_encoder", 150, "Size of the hidden states") #150 for bidaf ; #200 otherwise
tf.app.flags.DEFINE_integer("hidden_size_qp_matching", 150, "Size of the hidden states")
tf.app.flags.DEFINE_integer("hidden_size_sm_matching", 50, "Size of the hidden states")
tf.app.flags.DEFINE_integer("hidden_size_fully_connected", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 300, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")


## Bool flags to select different models
tf.app.flags.DEFINE_bool("do_char_embed", False, "Include char embedding -True/False")
tf.app.flags.DEFINE_bool("add_highway_layer", True, "Add highway layer to concatenated embeddings -True/False")
tf.app.flags.DEFINE_bool("cnn_encoder", False, "Add CNN Encoder Layer -True/False")
tf.app.flags.DEFINE_bool("rnet_attention", False, "Perform RNET QP and SM attention-True/False")
tf.app.flags.DEFINE_bool("bidaf_attention", True, "Use BIDAF Attention-True/False")
tf.app.flags.DEFINE_bool("answer_pointer_RNET", False, "Use Answer Pointer from RNET-True/False")
tf.app.flags.DEFINE_bool("smart_span", True, "Select start and end idx based on smart conditions-True/False")

## Hyperparameters for Char CNN
tf.app.flags.DEFINE_integer("char_embedding_size", 8, "Size of char embedding")  #as suggested in handout
tf.app.flags.DEFINE_integer("word_max_len", 16, "max length for each word") # 99th percentile from Jupyter notebook
tf.app.flags.DEFINE_integer("char_out_size", 100, "num filters char CNN/out size") # same as filer size; as suggested in handout
tf.app.flags.DEFINE_integer("window_width", 5, "Kernel size for char cnn") #as suggested in handout


## Hyperparameters for CNN Encoder
tf.app.flags.DEFINE_integer("filter_size_encoder", 20, "Size of filter for cnn encoder")

## Hyperparameters for BIDAF
tf.app.flags.DEFINE_integer("hidden_size_modeling", 150, "Size of modeling layer")  #

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "../experiments/bidaf_best/best_checkpoint", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "../data/tiny-dev-test.json", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "../predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


import random
import json
import numpy as np

def generate_answers():
	dummy_answers_list = []
	for i in range(np.random.randint(2,4)):
		dummy_answers_list.append({'answer_start':np.random.randint(0,100), 'text':'Just random stuff to stump you'})
	return dummy_answers_list


def generate_qas(questions_list):
	qas_list = []
	for ques in questions_list:
		rand_id = ''.join(random.choice('0123456789abcde') for i in range(24))
		qas_list.append({'answers':generate_answers(), 'id':rand_id, 'question':ques})
	return qas_list


def generate_context(context, questions_list):
	return {'context':context, 'qas':generate_qas(questions_list)}

def generate_multiple_context(context, questions_list):
	return [generate_context(ctx,questions_list) for ctx in context]


def generate_paragraphs(title, context, questions_list):
	return [{'title':title, 'paragraphs':[generate_context(context,questions_list)]}]

def generate_multiple_paragraphs(title, context, questions_list):
	return [{'title':title, 'paragraphs':generate_multiple_context(context,questions_list)}]

def generate_json(version, title, context, questions_list):
	return {'version':version, 'data':generate_paragraphs(title, context, questions_list)}

def generate_multiple_json(version, title, context, questions_list):
	return {'version':version, 'data':generate_multiple_paragraphs(title, context, questions_list)}


# def context_parser(text):
#     string_list = text.split("Question--")
#     context = re.sub(r'context -','', string_list[0]).strip()
#     question = re.sub(r'^ -','', string_list[1]).strip()
#     return(context, [question])

def context_parser(text_in):
	c_status = True if re.search('context', text_in, re.IGNORECASE) else False
	q_certain =  True if re.search('question', text_in, re.IGNORECASE) else False
	q_probable = True if re.search('\?|What|Who|How|When|Which|Where', text_in, re.IGNORECASE) else False
	get_context = True if re.search('(get|current) context', text_in, re.IGNORECASE) else False
	insofe_filter = re.compile(re.escape('in Sophie'), re.IGNORECASE)
	text_in = insofe_filter.sub('INSOFE', text_in)
	print('c_status',c_status)
	if get_context:
		text_type = 'GC'
		text_out = text_in
	elif len(text_in)<10:
		text_type = 'O'
		text_out = "I'd love to have a humane conversation, but right now I'm just designed to take a context and answer questions based on that. Maybe next time..."
	elif c_status or len(text_in) > 100:
		text_type = 'C'
		text_out = text_in
	elif q_certain:
		text_type = 'Q'
		pattern = re.compile("question", re.IGNORECASE)
		text_out = pattern.sub("",text_in).strip()
	elif q_probable and len(text_in) < 100:
		text_type = 'Q'
		text_out = text_in
	return(text_type, [text_out])

def initialize_model(session, model, train_dir, expect_exists):
	"""
	Initializes model from train_dir.

	Inputs:
	  session: TensorFlow session
	  model: QAModel
	  train_dir: path to directory where we'll look for checkpoint
	  expect_exists: If True, throw an error if no checkpoint is found.
		If False, initialize fresh model if no checkpoint is found.
	"""
	print "Looking for model at %s..." % train_dir
	ckpt = tf.train.get_checkpoint_state(train_dir)
	v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
	if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
		print "Reading model parameters from %s" % ckpt.model_checkpoint_path
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		if expect_exists:
			raise Exception("There is no saved checkpoint at %s" % train_dir)
		else:
			print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
			session.run(tf.global_variables_initializer())
			print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())


def main(unused_argv):
	# Print an error message if you've entered flags incorrectly


	if len(unused_argv) != 1:
		raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

	# Check for Python 2
	if sys.version_info[0] != 2:
		raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

	# Print out Tensorflow version
	print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

	# Define train_dir
	if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval":
		raise Exception("You need to specify either --experiment_name or --train_dir")
	FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

	# Initialize bestmodel directory
	global bestmodel_dir
	bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")
	
	# Define path for glove vecs
	FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

	# Load embedding matrix and vocab mappings
	global emb_matrix, word2id, id2word
	emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)
	
	# Initialize model
	global qa_model
	qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)

	# Some GPU settings
	global config
	config=tf.ConfigProto()
	config.gpu_options.allow_growth = True

	global sess
	sess =  tf.Session(config=config)

	global global_context
	global_context = 'INSOFE has awarded over Rs 3.2 Crores in merit scholarships in the last 2 years alone. INSOFE recognizes top performers and rewards them for demonstrating outstanding achievement at every phase of the program based on their performance and eligibility criteria. At each phase of the program, top performers are awarded rankings based on which scholarship winners are announced. Top performers can potentially win scholarships ranging from Rs 25,000 to entire program fee and this can be attained on the successful completion of the program.'
	global global_context_list
	global_context_list = ['INSOFE has awarded over Rs 3.2 Crores in merit scholarships in the last 2 years alone. INSOFE recognizes top performers and rewards them for demonstrating outstanding achievement at every phase of the program based on their performance and eligibility criteria. At each phase of the program, top performers are awarded rankings based on which scholarship winners are announced. Top performers can potentially win scholarships ranging from Rs 25,000 to entire program fee and this can be attained on the successful completion of the program.',
	'INSOFE is working on developing a video surveillance tool with enhanced smart capabilities. The tool identifies the violation and sends out instant automated response without requiring any manual interference. Since the current process involves manually going through the footage and checking for violations, it is not only a time-consuming process but also requires manual hours and effort. The tool makes the entire process automated with an Embedded Machine Learning chip Question',
	'Dr Dakshinamurthy, is the Founder and President of INSOFE. He did his PhD in Materials Science and Engineering from Carnegie Mellon University. He is known for simplifying complex ideas and communicating them clearly and excitingly. Dr Sridhar Pappu is the Executive VP - Academics of INSOFE. He leads the academic administration of the institute and ensures the highest standards in learning for the students. He teaches statistics. He loves data, soo much that he wears two fitness trackers.']
	
	# Load model from ckpt_load_dir
	initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)
	

	# app.run(host='0.0.0.0', port=443, ssl_context=('/home/gem/.ssh/certificate.pem', '/home/gem/.ssh/private-key.pem'))
	app.run(debug=True, host='0.0.0.0', port=443, ssl_context=('/etc/letsencrypt/live/gem.eastus2.cloudapp.azure.com/fullchain.pem', '/etc/letsencrypt/live/gem.eastus2.cloudapp.azure.com/privkey.pem'))


# fpath = '/home/sagarp/gem/cs224n-Squad-Project/data/tiny-dev-test.json'
# initialize our Flask application and the Keras model
app = flask.Flask(__name__, template_folder='../static', static_folder='../static')
CORS(app)
ask = Ask(app, "/")

@app.route("/predict", methods=["GET", "POST"])
def predict():
	# initialize the data dictionary that will be returned from the view
	data = {"success": False}
	print("Predict call")
	# ensure an text was properly uploaded to our endpoint
	if flask.request.method == "POST":
		print('Got request')
		print('values',flask.request.values)
		print('get_json',flask.request.get_json())
		# print('flask value',flask.request.values)
		if flask.request.values:
			message = flask.request.values['ticket']
			text_type, new_text = context_parser(message)
			print('Text type',text_type)
			print('Global context',global_context)
			print('New text',new_text)

			if text_type=="GC":
				data["predictions"] = {"answer": "Current context -- " + global_context, "probability": 1}
				print('Get context')

			elif text_type=="O":
				data["predictions"] = {"answer": new_text[0], "probability": 1}
				print('Other')

			elif text_type=='Q':
				print('Question')
				print(new_text)
				new_json = generate_json(version=1, title="INSOFE", context=global_context, questions_list=new_text)

				# with open(fpath, 'w') as outfile:
				#     json.dump(new_json, outfile)

				# Read the JSON data from file
				# qn_uuid_data, context_token_data, qn_token_data = get_json_data(fpath)
				qn_uuid_data, context_token_data, qn_token_data = get_json_data(new_json)
				answers_dict = generate_answers_prob(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)
				print(answers_dict, 'ad')
				# print(answers_dict.values(), 'ad')

				# Write the uuid->answer mapping a to json file in root dir
				# print "Writing predictions to %s..." % FLAGS.json_out_path
				# with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
				# 	f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
				# 	print "Wrote predictions to %s" % FLAGS.json_out_path
				
				data["predictions"] = {"answer": answers_dict.values()[0][0], "probability": answers_dict.values()[0][1]}
				#  = r
				# print(r)
				# indicate that the request was a success
				data["success"] = True
			elif text_type=='C':
				global global_context
				global_context = new_text[0]
				data["predictions"] = {"answer": 'Context updated', "probability": 1}

	else:
		print('Incorrect request type')
	
	# return the data dictionary as a JSON response
	return flask.jsonify(data)


@app.route('/hello', methods=['GET'])
def upload_file():
	return '''
	<!doctype html>
	<title>HomePage</title>
	<h1>Hello from The Dude</h1>
	</form>
	'''


@app.route("/")
def hello():  
    return render_template('index.html', )


@ask.launch
def welcome():
	print('welcome')
	welcome_msg = render_template('welcome')
	return question(welcome_msg)


@ask.intent("answer")
def answer():
	print('Answering mode')
	alexa_question = ask_request.intent.slots.question.value
	print(alexa_question)
	text_type, new_text = context_parser(alexa_question)
	print('Text type',text_type)
	print('Global context',global_context)
	print('New text',new_text)
	new_json = generate_multiple_json(version=1, title="INSOFE", context=global_context_list, questions_list=new_text)
	print(new_json)
	if text_type=='Q':
		qn_uuid_data, context_token_data, qn_token_data = get_json_data(new_json)
		answers_dict = generate_answers_prob(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)
		answers_dict = sorted(answers_dict.items(), key=lambda e: e[1][1], reverse=True)
		print(answers_dict, 'ad')
		# answer = answers_dict.values()[0][0].replace('insofe','insofee')
		answer = answers_dict[0][1][0].replace('insofe','insofee')
	else:
		answer = "I'm not sure if I understand. Come again"
	return question(answer)


@ask.intent('AMAZON.StopIntent')
def stop():
	bye_text = render_template('bye')
	return statement(bye_text)


@ask.session_ended
def session_ended():
	end_text = render_template('sorry')
	return question(end_text)


@ask.intent('josh')
def josh():
	josh_text = render_template('josh')
	return question(josh_text)


@ask.intent('gem')
def gem():
	gem_text = render_template('gem')
	return question(gem_text)


@ask.intent('AMAZON.FallbackIntent')
def fallback():
	fallback_text = render_template('sorry')
	return question(fallback_text)


if __name__ == "__main__":
	tf.app.run()
