#-*- coding: UTF-8 -*-
import math
import os
import json
import numpy as np

TRAIN_SIZE = 45000

punc = ['（', '）', '，', '。', '：', '！', '？', '、']

feature = dict()
fcnt = 0

def is_valid_word(word):
	return len(word) > 0 and len(word) * 3 == len(word.encode('utf-8')) and not (word in punc)

def read_words(link):
	f = open(link)
	lines = f.readlines()
	words = []
	for line in lines:
		ws = line.strip().split(' ')
		for w in ws:
			if (is_valid_word(w)):
				words.append(w)
	return words


def load_email_feature(link):
	global feature

	feat = np.zeros(len(feature))

	words = read_words(link)

	for word in words:
		if (word in feature):
			f = feature[word][1]
			feat[f] += 1

	return feat

def load_email(kind, link, prob_y, prob_f):
	global feature
	isham = 1 if kind == "ham" else 0
	prob_y[isham] += 1


	words = read_words(link)

	for word in words:
		if (word in feature):
			f = feature[word][1]
			prob_f[isham][f] += 1
			feat[f] += 1

	return feat

def load_feature(kind, link):
	global feature
	global fcnt
	isspam = 1 if kind == "spam" else 0

	words = read_words(link)
	for word in words:
		if (word not in feature):
			feature[word] = [isspam, fcnt, word]
			fcnt += 1
		else:
		 	feature[word][0] += isspam

def load_dict(fname):
	with open(fname, 'r') as json_file:
		dic = json.load(json_file)
	return dic

def save_dict(fname, dic):
	with open(fname, 'w') as json_file:
		json.dump(dic, json_file, ensure_ascii = False)

def load_data():
	global feature
	global fcnt
	flabel = open("../data/label/index")
	emails = flabel.readlines()[0:TRAIN_SIZE]

	if (os.path.exists("feature.dat")):
		feature = load_dict("feature.dat")
		print(feature)
		fcnt = len(feature)
	else:
		for email in emails:
			kind = email.split(' ')[0]
			link = "../data/data_cut/" + email.split(' ')[1].strip()[-7:]
			print(link)
			load_feature(kind, link)

		fe = [ v for v in sorted(feature.values(), reverse = True)]
		feature = dict()
		for i in range(0, 1000):
			feature[fe[i][2]] = [fe[i][0], i]

		fcnt = 1000

		save_dict("feature.dat", feature)


	if (os.path.exists("prob_f.txt") and os.path.exists("prob_y.txt")):
		prob_f = np.loadtxt("prob_f.txt")
		prob_y = np.loadtxt("prob_y.txt")
	else:
		prob_y = np.array([0, 0])
		prob_f = np.ones([2, fcnt])
		for email in emails:
			kind = email.split(' ')[0]
			link = "../data/data_cut/" + email.split(' ')[1].strip()[-7:]
			load_email(kind, link, prob_y, prob_f)
			print(link)

		prob_f[0] = prob_f[0] / prob_y[0]
		prob_f[1] = prob_f[1] / prob_y[1]
		prob_y = prob_y / prob_y.sum()
		np.savetxt('prob_y.txt', prob_y)
		np.savetxt('prob_f.txt', prob_f)

	return (prob_y, prob_f)

class BayesClassifier():
	#(1 * y_cnt)
	#(y_cnt * f_cnt)
	def __init__(self, prob_y, prob_f):
		self.prob_y = prob_y
		self.prob_f = prob_f

	def classify(self, x):
		y = self.prob_y
		f = self.prob_f
		prob = np.matmul(x, np.log(f).T)
		return "ham" if np.argmax(prob) else "spam"

if __name__ == '__main__':

	prob_y, prob_f = load_data()
	print(prob_y)
	print(prob_f)
	c = BayesClassifier(prob_y, prob_f)

	flabel = open("../data/label/index")
	emails = flabel.readlines()[TRAIN_SIZE:]

	TP = 0
	FP = 0
	FN = 0
	TN = 0
	for email in emails:
		link = "../data/data_cut/" + email.split(' ')[1].strip()[-7:]
		kind = email.split(' ')[0]
		x = load_email_feature(link)
		res = c.classify(x)
		if (kind == "spam" and res == "spam"):
			TP += 1
		if (kind == "spam" and res == "ham"):
			FN += 1
		if (kind == "ham" and res == "spam"):
			FP += 1
		if (kind == "ham" and res == "ham"):
			TN += 1

		accuracy = (TP + TN) / (TP + TN + FP + FN)
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		mesure = 2 * precision * recall / (precision + recall)


		print("Accuracy: %.3f Precision %.3f Recall %.3f F1-Measure %.3f" % (accuracy, precision, recall, mesure))


	print(correct, '/', len(emails), "%.3f" % (correct / len(emails)))
