import gpflow
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from toy import rbf_toy2, rbf_toy, x3_toy, x3_gap_toy, sin_toy, comp_toy
import util

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='x3')
parser.add_argument('--dir_out', type=str, default='output/')
parser.add_argument('--fixed_standarization', type=bool, default=True, \
    help='whether to use same data standardization constants for each dataset')
parser.add_argument('--opt_kernel_hyperparam', type=bool, default=False)
parser.add_argument('--opt_likelihood_variance', type=bool, default=False)
args = parser.parse_args()

## data
dataset = dict(rbf=rbf_toy2, x3=x3_toy, sin=sin_toy, comp=comp_toy)[args.dataset]()
n_train_all = np.arange(10,20,10)
#n_train_all = np.arange(20,220,20) # dataset sizes to run
#n_train_all = np.array([250])
n_rep = 10
seed = 0
n_test = 1000

## allocate space
n_all = n_train_all.size
rmse = np.zeros((n_all, n_rep))
picp = np.zeros((n_all, n_rep))
mpiw = np.zeros((n_all, n_rep))
test_log_likelihood = np.zeros((n_all, n_rep))

if not os.path.exists(args.dir_out):
    os.makedirs(args.dir_out)

## kernel
if dataset.name == 'rbf':
    kern = gpflow.kernels.RBF(input_dim=1, lengthscales=1.0, variance=1.0)

elif dataset.name == 'x3':
    kern = gpflow.kernels.Polynomial(input_dim=1, degree=3, variance=1.0, offset=0.0)

elif dataset.name == 'sin':
    #kern = gpflow.kernels.Periodic(input_dim=1) + gpflow.kernels.RBF(input_dim=1)
    kern = gpflow.kernels.Periodic(input_dim=1, lengthscales=0.002, variance=2.2, period=273.4) + gpflow.kernels.RBF(input_dim=1, lengthscales=31.6, variance=1.0)

elif dataset.name == 'comp':
    #kern = gpflow.kernels.Polynomial(degree=1, input_dim=1) * gpflow.kernels.Periodic(input_dim=1)
    kern = gpflow.kernels.Polynomial(degree=1, input_dim=1, offset=1.11, variance=1.58) * \
           gpflow.kernels.Periodic(input_dim=1, lengthscales=1.20, variance=1.33, period=1.31)

if dataset.name == 'rbf':
    dataset.sample_f(n_train_max=np.maximum(1000, np.max(n_train_all)), n_test=n_test, seed=seed) # initialize new function

if args.fixed_standarization:
    x_standard, y_standard = dataset.train_samples(n_data=1000, seed=0)
    mean_x_standard, std_x_standard = np.mean(x_standard), np.std(x_standard)
    mean_y_standard, std_y_standard = np.mean(y_standard), np.std(y_standard)


for i, n_train in enumerate(n_train_all):

    for j in range(n_rep):
        seed += 2

        ## data
        original_x_train, original_y_train = dataset.train_samples(n_data=n_train, seed=seed)
        original_x_test, original_y_test = dataset.test_samples(n_data=n_test, seed=seed+1)

        # standarize data
        if args.fixed_standarization:            
            train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                     original_x_test, original_y_test,
                                                                     mean_x_standard, std_x_standard, mean_y_standard, std_y_standard) 
            noise_std = dataset.y_std / std_y_standard

        else:
            train_x, train_y, test_x, test_y = util.standardize_data(original_x_train, original_y_train, \
                                                                     original_x_test, original_y_test)
            noise_std = dataset.y_std / np.std(original_y_train)

        #plt.plot(test_x, test_y)
        #plt.savefig('temp.png')

        ## model
        m = gpflow.models.GPR(train_x, train_y, kern=kern)

        # likelihood variance
        if not args.opt_likelihood_variance:
            m.likelihood.variance.trainable = False
            m.likelihood.variance = noise_std**2 # fixed observation variance

        # optimize hyperparameters
        if args.opt_kernel_hyperparam:
            #opt = gpflow.train.ScipyOptimizer() # Replace with AdamOptimizer?
            #opt.minimize(m)
            opt = gpflow.train.AdamOptimizer(.05)
            opt.minimize(m)

        noise_std_opt = m.likelihood.variance.read_value().item()
        print('noise: ', noise_std_opt)

        print(m.kern.as_pandas_table())

        ## posterior

        # plot
        xx = np.linspace(-2, 2, 100).reshape(100, 1)  # test points must be of shape (N, D)
        mean, var = m.predict_y(xx)
        samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)
        plt.figure(figsize=(12, 6))
        plt.plot(train_x, train_y, 'kx', mew=2)
        #plt.plot(test_x, test_y, 'rx', mew=2)
        plt.plot(xx, mean, 'C0', lw=2)
        plt.fill_between(xx[:,0],
                         mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                         mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                         color='C0', alpha=0.2)
        plt.ylim(-3,3)
        plt.savefig(args.dir_out + '/posterior_n_train=%d_rep=%d.png' % (n_train, j))
        plt.close()

        # functions
        f_pred, f_pred_cov = m.predict_f_full_cov(test_x)
        f_pred_cov = f_pred_cov[0]
        f_pred_var = np.diagonal(f_pred_cov).reshape(-1,1)

        f_pred_lb = f_pred - 1.96 * np.sqrt(f_pred_var)
        f_pred_ub = f_pred + 1.96 * np.sqrt(f_pred_var)

        # predictive (i.e. including noise variance)
        y_pred_cov = f_pred_cov + noise_std_opt*np.eye(f_pred.shape[0]) 
        y_pred, y_pred_var = m.predict_y(test_x) # y_pred = f_pred

        y_pred_lb = y_pred - 1.96 * np.sqrt(y_pred_var)
        y_pred_ub = y_pred + 1.96 * np.sqrt(y_pred_var)

        ## evaluate posterior
        #y_pred_cov = np.diag(np.diagonal(y_pred_cov)) # TEMP use diagonal of covariance
        test_log_likelihood[i,j] = util.test_log_likelihood(y_pred, y_pred_cov, test_y)
        rmse[i,j] = util.rmse(test_y, y_pred)
        picp[i,j] = util.picp(test_y, y_pred_lb, y_pred_ub)
        mpiw[i,j] = util.mpiw(y_pred_lb, y_pred_ub)
    

## plot
fig, ax = plt.subplots(1,4,figsize=(16, 4))

ax[0].set_title('log likelihood')
#ax[0].errorbar(n_train_all, np.mean(test_log_likelihood, 1), np.std(test_log_likelihood, 1), None, '-o')
ax[0].boxplot(test_log_likelihood.T, positions=n_train_all)
ax[0].plot(n_train_all, np.mean(test_log_likelihood, 1))

ax[1].set_title('rmse')
#ax[1].errorbar(n_train_all, np.mean(rmse, 1), np.std(rmse, 1), None, '-o')
ax[1].boxplot(rmse.T, positions=n_train_all)
ax[1].plot(n_train_all, np.mean(rmse, 1))

ax[2].set_title('prediction interval coverage (PICP)')
#ax[2].errorbar(n_train_all, np.mean(picp, 1), np.std(picp, 1), None, '-o')
ax[2].boxplot(picp.T, positions=n_train_all)
ax[2].plot(n_train_all, np.mean(picp, 1))

ax[3].set_title('mean prediction interval width (MPIW)')
#ax[3].errorbar(n_train_all, np.mean(mpiw, 1), np.std(mpiw, 1), None, '-o')
ax[3].boxplot(mpiw.T, positions=n_train_all)
ax[3].plot(n_train_all, np.mean(mpiw, 1))

fig.savefig(args.dir_out + '/results.png')
plt.close()
