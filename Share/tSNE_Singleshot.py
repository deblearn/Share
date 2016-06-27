#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as Math
import pylab as Plot

def Hbeta(D = Math.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = Math.exp(-D.copy() * beta);
	sumP = sum(P);
	H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = Math.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
	sum_X = Math.sum(Math.square(X), 1);
	D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
	P = Math.zeros((n, n));
	beta = Math.ones((n, 1));
	logU = Math.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -Math.inf;
		betamax =  Math.inf;
		Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while Math.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == Math.inf or betamax == -Math.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == Math.inf or betamin == -Math.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print "Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta));
	return P;


def pca(X = Math.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - Math.tile(Math.mean(X, 0), (n, 1));
	(l, M) = Math.linalg.eig(Math.dot(X.T, X));
	Y = Math.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
		print "Error: array X should have type float.";
		return -1;
	if round(no_dims) != no_dims:
		print "Error: number of dimensions should be an integer.";
		return -1;

	# Initialize variables
	X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = Math.random.randn(n, no_dims);
	dY = Math.zeros((n, no_dims));
	iY = Math.zeros((n, no_dims));
	gains = Math.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	#(bbb, bbbb) = P.shape;
	P = P + Math.transpose(P);
	P = P / Math.sum(P);
	P = P * 4;									# early exaggeration
	P = Math.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1);
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / Math.sum(num);
		Q = Math.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		ppp = Math.tile(Math.mean(Y, 0), (n, 1));
		Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
		# print(Y[1][0])
		# print(Y[2][1])

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = Math.sum(P * Math.log(P / Q));
			print "Iteration ", (iter + 1), ": error is ", C

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;


def tsne1(Shared_length, Site_length, X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0, Y_1 = Math.array([])):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	def updateS(Y,G):
		return Y_1

	def updateL(Y,G):
		return Y + G

	def demeanS(Y):
		return Y

	def demeanL(Y):
		return Y - Math.tile(Math.mean(Y, 0), (Y.shape[0], 1))

	# Check inputs
	if isinstance(no_dims, float):
		print "Error: array X should have type float.";
		return -1;
	if round(no_dims) != no_dims:
		print "Error: number of dimensions should be an integer.";
		return -1;

	# Initialize variables
	#Y = [[0 for i in range(1778)] for j in range(2)]
	difference = Site_length- Shared_length;
	print(Shared_length, Site_length,difference)

	Y = Math.random.randn(Site_length, no_dims);
	X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	index=0;
	index1=0;
	Y_2 = Math.random.randn(difference, no_dims);

	Y[:Shared_length,:] = Y_1
	Y[Shared_length:,:] = Y_2

	dY = Math.zeros((n, no_dims));
	iY = Math.zeros((n, no_dims));
	gains = Math.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	#(bbb, bbbb) = P.shape;
	P = P + Math.transpose(P);
	P = P / Math.sum(P);
	P = P * 4;									# early exaggeration
	P = Math.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1);
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / Math.sum(num);
		Q = Math.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);

		Y[:Shared_length,:] = updateS(Y[:Shared_length,:], iY[:Shared_length,:])
		Y[Shared_length :, :] = updateL(Y[Shared_length:,:], iY[Shared_length:,:])

		#p=0;
		Y[:Shared_length,:] = demeanS(Y[:Shared_length,:])
		Y[Shared_length :, :] = demeanL(Y[Shared_length:,:])

		#Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
		#print(Y[1][0])
		#print(Y[2][1])

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = Math.sum(P * Math.log(P / Q));
			print "Iteration ", (iter + 1), ": error is ", C

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;

if __name__ == "__main__":
	print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
	print "Running example on 2,500 MNIST digits..."


	# X = Math.loadtxt("mnist2500_X.txt");
	# labels = Math.loadtxt("mnist2500_labels.txt");

	Combined_X = Math.loadtxt("Preprocessing_Mnist_X.txt");
	Combined_labels = Math.loadtxt("Preprocessing_label.txt");

	X = Math.loadtxt("Shared_Mnist_X.txt");
	labels = Math.loadtxt("Shared_lable.txt");
	Site_1 = Math.loadtxt("Site_1_Mnist_X.txt");
	labels_1 = Math.loadtxt("Site_1_Lable.txt");
	Site_2 = Math.loadtxt("Site_2_Mnist_X.txt");
	labels_2 = Math.loadtxt("Site_2_Lable.txt");
	Site_3 = Math.loadtxt("Site_3_Mnist_X.txt");
	labels_3 = Math.loadtxt("Site_3_Lable.txt");

	(Shared_length_X, Shared_length_Y) = X.shape;
	(Site1_length_X, Site1_length_Y) = Site_1.shape;
	(Site2_length_X, Site2_length_Y) = Site_2.shape;
	(Site3_length_X, Site3_length_Y) = Site_3.shape;

	print(Shared_length_X, Shared_length_Y)
	print(Site1_length_X, Site1_length_Y)

	#Y_Combined = tsne(Combined_X, 2, 20, 50.0);

	Y11 = tsne(X, 2, 50, 30.0);
	Y22 = tsne1(Shared_length_X, Site1_length_X, Site_1, 2, 50, 30.0, Y11);
	Y23 = tsne1(Shared_length_X, Site2_length_X, Site_2, 2, 50, 30.0, Y11);
	Y24 = tsne1(Shared_length_X, Site3_length_X, Site_3, 2, 50, 30.0, Y11);

	f = open("shared1.txt", "w")
	for i in range(0, Shared_length_X):
		f.write(str(Y11[i][0]) + '\t')  # str() converts to string
		f.write(str(Y11[i][1]) + '\n')  # str() converts to string

	f.close()
	f1 = open("site1.txt", "w")
	for i in range(0, Site1_length_X):
		f1.write(str(Y22[i][0]) + '\t')  # str() converts to string
		f1.write(str(Y22[i][1]) + '\n')  # str() converts to string

	f1.close()
	f2 = open("site2.txt", "w")
	for i in range(0, Site2_length_X):
		f2.write(str(Y23[i][0]) + '\t')  # str() converts to string
		f2.write(str(Y23[i][1]) + '\n')  # str() converts to string

	f2.close()
	f3 = open("site3.txt", "w")
	for i in range(0, Site3_length_X):
		f3.write(str(Y24[i][0]) + '\t')  # str() converts to string
		f3.write(str(Y24[i][1]) + '\n')  # str() converts to string

	f3.close()

	print(Y11[0][0])
	print(Y24[0][0])
	print(Y11[2][1])
	print(Y24[2][1])
	print(Y11[1261][0])
	print(Y24[1261][0])
	print(Y11[1262][1])
	print(Y24[1262][1])
	print(Y11[1263][0])
	print(Y24[1263][0])
	print(Y11[1264][1])
	print(Y24[1264][1])
	print(Y24[1265][0])
	print(Y11[1265][0])
	print(Y24[1265][1])
	print(Y11[1265][1])
	print(Y24[1266][0])
	print(Y24[1267][0])
	print(Y24[1268][1])

	Plot.scatter(Y11[:, 0], Y11[:, 1], 20, labels);
	Plot.scatter(Y22[:, 0], Y22[:, 1], 20, labels_1);
	Plot.scatter(Y23[:, 0], Y23[:, 1], 20, labels_2);
	Plot.scatter(Y24[:, 0], Y24[:, 1], 20, labels_3);
	#Plot.scatter(Y_Combined[:, 0], Y_Combined[:, 1], 20, Combined_labels);
	Plot.show();
