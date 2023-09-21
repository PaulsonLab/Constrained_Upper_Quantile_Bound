import pandas as pd
import numpy
import matplotlib.pyplot as plt

# initial stats
eps = 1e-2
replications = 10
max_iter = 100
UQB_sol = 1
other_sol = 7
problems = 10

solver_list = ['uqb', 'eicf','eic', 'epbo','snobfit', 'cmaes', 'bobyqa', 'direct']


import sys

def calculate_regret(f_data,maximize=False):
    for i in range(len(f_data)):
        if i == 0:
            regret = numpy.array([f_data[i]])
        else:
            if maximize is False:
                if f_data[i] < regret[-1]:
                    regret = numpy.append(regret, f_data[i])
                else:
                    regret = numpy.append(regret, regret[-1])
            else:
                if f_data[i] > regret[-1]:
                    regret = numpy.append(regret, f_data[i])
                else:
                    regret = numpy.append(regret, regret[-1])
    return regret

# convert UQB data
for i in range(problems):
    for j in range(len(solver_list)):
        sol = solver_list[j]
        for k in range(replications):
            if j == 0:
                r_data = pd.read_csv(f'Data/Unconstrained/R_problem-{i}_solver-7_rep-{k}_alpha-0.95_ninit-2.csv')
                r_data = r_data[:max_iter].to_numpy()[:,1:]
            else:
                r_data = pd.read_csv(f'Data/Unconstrained_solvers/R_problem-{i}_solver-{j-1}_rep-{k}.csv')
                
                if j > 3:
                    r_data = pd.read_csv(f'Data/Unconstrained_solvers/R_problem-{i}_solver-{j-1}_rep-{k}.csv')                      
                    r_data = r_data[:max_iter].to_numpy()[:,1:]
                    if i == 1:
                        r_data = -1*r_data
                else:
                    g_data = pd.read_csv(f'Data/Unconstrained_solvers/G_problem-{i}_solver-{j-1}_rep-{k}.csv')
                    g_ = g_data[:max_iter].to_numpy()[:,1:]
                    r_data = calculate_regret(g_,maximize=True)
            if len(r_data) < max_iter:
                best = r_data[-1]
                best_val = best*numpy.ones((max_iter - len(r_data),1))
                r_data = numpy.append(r_data, best_val)
            if k == 0:
                if r_data.ndim == 1:
                    rep_data = numpy.expand_dims(r_data,1)
                else:
                    rep_data = r_data
            else:
                if r_data.ndim == 1:
                    r_data = numpy.expand_dims(r_data,1)
                rep_data = numpy.append(rep_data, r_data, 1)      
        if j == 0:
            sol_data = [rep_data]
        else:
            sol_data.append(rep_data)
            
    # calculate median value
    for j in range(len(solver_list)):
         if j == 0:
            med_data = [numpy.median(sol_data[j],1)]
         else:
            med_data.append(numpy.median(sol_data[j],1))
    
    #calculate the max for median values:
    fmax = numpy.max(med_data)

    if i == 0:
        f_opt = [fmax]
    else:
        f_opt.append(fmax)
    
    # calculate performance data
    for j in range(len(solver_list)):
        f0 = med_data[j][0] - 1e-5
        m_data = (med_data[j] - f0)/(fmax-f0)
        
        if j == 0:
            perf_data = [m_data]
        else:
            perf_data.append(m_data)

    # finally store problem data
    if i == 0:
        performance_data = numpy.greater(perf_data, 1-eps).astype(int)
    else:
        performance_data = numpy.add(performance_data, numpy.greater(perf_data, 1-eps).astype(int))
        
percentage_data = performance_data/10

x_iter = numpy.linspace(1,100,100)
skip=5
plt.plot(x_iter,percentage_data[0],label='uqb',marker='o',markevery=skip,linestyle='--')
plt.plot(x_iter,percentage_data[1],label='eic-cf',marker='v',markevery=skip,linestyle='-.')
plt.plot(x_iter,percentage_data[2],label='eic',marker='^',markevery=skip,linestyle=':')
plt.plot(x_iter,percentage_data[3],label='epbo',marker='x',markevery=skip,linestyle='--')
plt.plot(x_iter,percentage_data[4],label='snobfit',marker='1',markevery=skip,linestyle='-.')
plt.plot(x_iter,percentage_data[5],label='cmaes',marker='s',markevery=skip,linestyle=':')
plt.plot(x_iter,percentage_data[6],label='bobyqa',marker='p',markevery=skip,linestyle='--')
plt.plot(x_iter,percentage_data[7],label='direct',marker='*',markevery=skip,linestyle='-.')
lgd = plt.legend(bbox_to_anchor=(1.05,1.0),loc='upper left')
plt.ylabel('fraction of problems')
plt.xlabel('evaluations')
plt.ylim([0,1])
plt.xlim([0,100])
#plt.savefig('Unconstrained_all_solvers.png', dpi=600,bbox_extra_artists=(lgd,), bbox_inches='tight')
