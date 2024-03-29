import argparse
from toy.util.data import basic_data_instance
from toy.util.general import set_seed, device, evaluate, clean, get_weight_norms, compute_factors, est_EE, est_EE_cond
from toy.util.swag import *
from toy.util.model import StochasticMLP
import torch
import os
from datetime import datetime
from dnn.swag_repo.MI.util import to_bits, create_dataframe
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

from collections import defaultdict
import scipy.stats as stats


args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--inds_start", type=int, default=0)
args_form.add_argument("--inds_end", type=int, default=216)

args_form.add_argument("--lr", type=float, default=1e-2)
args_form.add_argument("--MI_const", type=float, default=1.5)
args_form.add_argument("--batch_sz", type=int, default=64)
args_form.add_argument("--lamb_init", type=float, default=0.)

args_form.add_argument("--epochs", type=int, default=300)
args_form.add_argument("--swa_start", type=int, default=200)
args_form.add_argument("--out_dir", type=str, default="results")

args_form.add_argument("--compute_MI_theta_D", default=False, action="store_true")
args_form.add_argument("--compute_MI_theta_D_only_start", type=int, default=0)
args_form.add_argument("--compute_MI_theta_D_only_end", type=int, default=24)

args_form.add_argument("--results", default=False, action="store_true")
args_form.add_argument("--data_only", default=False, action="store_true")

args = args_form.parse_args()
print(args)

print(args.out_dir)

data_instances = list(range(3))

archs = [[2, 256, 256, 128, 128, 5],
[2, 128, 128, 64, 64, 5],
[2, 64, 64, 32, 32, 5],
[2, 32, 32, 16, 16, 5],
]

decays = [0.0, 1e-2, 1e-1]

lamb_lrs = [0.0, 5e-4]

seeds = [0, 1, 2]


print("Num models:")
num_models = len(data_instances) * len(seeds) * len(lamb_lrs) * len(archs) * len(decays)
print(num_models)

plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('font', size=15)

# s -> S
# \mathbf{S} -> S'
vnames_pretty = {

    # repr
    "MI_mcs": r"$\hat{I}(X; Z_l^S)$",
    "MI_cond_mcs": r"$\hat{I}(X; Z_l^S | Y)$",

    "MI_jensens": r"$\breve{I}(X; Z_l^S)$",
    "MI_cond_jensens": r"$\breve{I}(X; Z_l^S | Y)$", # this one

    "EE_loss": r"$H(L^w)$",
    "EE_acc": r"$H(L^w)$",
    "EE_cond_loss": r"$H(L^w | Y)$",
    "EE_cond_acc": r"$H(L^w | Y)$",

    # model
    "MI_theta_singles": r"$\breve{I}({S'}; \theta_l^{S'})$", # jensen upper bound this one
    "MI_theta_multis": r"$\bar{I}({S'}; \theta_l^{S'})$", # double jensen bound

    "MI_theta_singles_last": r"$\breve{I}({S'}; \theta_{D+1}^{S'})$",
    "MI_theta_multis_last": r"$\bar{I}({S'}; \theta_{D+1}^{S'})$",

    # combined
    "combined_singles_mc": r"$\tilde{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S)$",
    "combined_single_conds_mc": r"$\tilde{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S | Y)$",

    "combined_singles_jensen": r"$\tilde{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S)$",
    "combined_single_conds_jensen": r"$\tilde{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S | Y)$", # this one

    # noscale
    "combined_singles_mc_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S)$",
    "combined_single_conds_mc_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \hat{I}(X; Z_l^S | Y)$",

    "combined_singles_jensen_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S)$",
    "combined_single_conds_jensen_noscale": r"$\breve{I}({S'}; \theta_l^{S'}) + \breve{I}(X; Z_l^S | Y)$",

    "num_params": r"Num. param. $m$",
    "VC": r"$m \log m$",
    "sum_weight_norms": r"$\sum_l \theta_l^\Sbb$",
    "prod_weight_norms": r"$\prod_l \theta_l^\Sbb$",

}


################################
# Training models
################################


model_ind = 0
plot_instance = [True] * len(data_instances)
for lamb_lr in lamb_lrs:
    for arch in archs:
        for decay in decays:

            for data_instance in data_instances:
                for seed in seeds:

                    if model_ind in range(args.inds_start, args.inds_end): # exclusive
                        results_p = os.path.join(args.out_dir, "results_%d.pt" % model_ind)

                        if args.data_only or os.path.exists(results_p):
                            print("Skipping %s" % results_p)
                            model_ind += 1
                            continue

                        train_dl, test_dl = basic_data_instance(args, args.batch_sz, data_instance, plot=plot_instance[data_instance])
                        plot_instance[data_instance] = False

                        print(("num batches", len(train_dl), len(test_dl)))

                        print("Doing model_ind %d, %s" % (model_ind, datetime.now()))
                        print((seed, lamb_lr, arch, decay))
                        sys.stdout.flush()

                        assert arch[-1] == args.C
                        set_seed(seed)

                        model = StochasticMLP(arch).to(device).train()
                        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=decay)

                        model, swag_model, diagnostics = train_model_swag(model, arch, opt,
                                                                          train_dl, test_dl, args,
                                                                          lamb_lr)

                        train_acc, train_loss = evaluate(model, train_dl, args, "train", plot=True)
                        test_acc, test_loss = evaluate(model, test_dl, args, "test", plot=True)

                        print("train: %.3f %.3f, test: %.3f %.3f, MI: mc %s %s jensen %s %s, diagnostics: test %.3f %.3f swag %.3f %.3f" %
                              (train_acc, train_loss, test_acc, test_loss,
                               diagnostics["MI_mc"], diagnostics["MI_cond_mc"], diagnostics["MI_jensen"], diagnostics["MI_cond_jensen"],
                               diagnostics["test_losses"][-1], diagnostics["test_accs"][-1],
                               diagnostics["test_loss_swag"], diagnostics["test_acc_swag"],
                               ))

                        # evaluate MI another time
                        MI_train_dl_5, _ = basic_data_instance(args, args.batch_sz, data_instance, size=5)

                        MI_mc_5 = est_MI(model, MI_train_dl_5.dataset,
                                             sz=len(MI_train_dl_5.dataset), jensen=False,
                                             requires_grad=False).item()
                        MI_cond_mc_5 = est_MI_cond(model, args.C, MI_train_dl_5,
                                                       sz=-1, jensen=False).item()

                        MI_jensen_5 = est_MI(model, MI_train_dl_5.dataset,
                                                 sz=len(MI_train_dl_5.dataset), jensen=True,
                                                 requires_grad=False).item()
                        MI_cond_jensen_5 = est_MI_cond(model, args.C, MI_train_dl_5,
                                                           sz=-1, jensen=True).item()

                        EE_loss_5 = est_EE(model, MI_train_dl_5, True)
                        EE_acc_5 = est_EE(model, MI_train_dl_5, False)
                        EE_cond_loss_5 = est_EE_cond(model, MI_train_dl_5, True, args.C)
                        EE_cond_acc_5 = est_EE_cond(model, MI_train_dl_5, False, args.C)

                        MI_train_dl_10, _ = basic_data_instance(args, args.batch_sz, data_instance, size=10)

                        MI_mc_10 = est_MI(model, MI_train_dl_10.dataset,
                                         sz=len(MI_train_dl_10.dataset), jensen=False,
                                         requires_grad=False).item()
                        MI_cond_mc_10 = est_MI_cond(model, args.C, MI_train_dl_10,
                                                   sz=-1, jensen=False).item()

                        MI_jensen_10 = est_MI(model, MI_train_dl_10.dataset,
                                             sz=len(MI_train_dl_10.dataset), jensen=True,
                                             requires_grad=False).item()
                        MI_cond_jensen_10 = est_MI_cond(model, args.C, MI_train_dl_10,
                                                       sz=-1, jensen=True).item()

                        EE_loss_10 = est_EE(model, MI_train_dl_10, True)
                        EE_acc_10 = est_EE(model, MI_train_dl_10, False)
                        EE_cond_loss_10 = est_EE_cond(model, MI_train_dl_10, True, args.C)
                        EE_cond_acc_10 = est_EE_cond(model, MI_train_dl_10, False, args.C)

                        # other factors
                        weight_norms, num_params = get_weight_norms(model)

                        VC = num_params * np.log2(num_params)

                        sum_weight_norms = weight_norms.sum().item()

                        prod_weight_norms = weight_norms.prod().item()

                        nplots = len(diagnostics)
                        fig, axarr = plt.subplots(nplots, figsize=(4, nplots * 4))
                        for plot_i, (plot_name, plot_values) in enumerate(diagnostics.items()):

                            axarr[plot_i].plot(plot_values)
                            axarr[plot_i].set_ylabel(plot_name)
                            if plot_i == 0:
                                axarr[plot_i].set_title("test acc %.3E loss %.3E"% (test_acc, test_loss))

                        plt.tight_layout()
                        fig.savefig(os.path.join(args.out_dir, "training_%d.pdf" % model_ind), bbox_inches="tight")
                        plt.close("all")

                        results = {
                            "model_ind": model_ind,

                            "swag_model": swag_model,
                            "model": model,

                            "arch": arch,
                            "decay": decay,
                            "lamb_lr": lamb_lr,
                            "seed": seed,

                            "train_acc": train_acc,
                            "train_loss": train_loss,
                            "test_acc": test_acc,
                            "test_loss": test_loss,

                            "gen_gap_err": train_acc - test_acc, # = 1 - test_acc - (1 - train_acc)
                            "gen_gap_loss": test_loss - train_loss,

                            "diagnostics": diagnostics
                        }

                        results["MI_mc_5"] = MI_mc_5
                        results["MI_cond_mc_5"] = MI_cond_mc_5
                        results["MI_jensen_5"] = MI_jensen_5
                        results["MI_cond_jensen_5"] = MI_cond_jensen_5

                        results["EE_loss_5"] = EE_loss_5
                        results["EE_acc_5"] = EE_acc_5
                        results["EE_cond_loss_5"] = EE_cond_loss_5
                        results["EE_cond_acc_5"] = EE_cond_acc_5

                        results["MI_mc_10"] = MI_mc_10
                        results["MI_cond_mc_10"] = MI_cond_mc_10
                        results["MI_jensen_10"] = MI_jensen_10
                        results["MI_cond_jensen_10"] = MI_cond_jensen_10

                        results["EE_loss_10"] = EE_loss_10
                        results["EE_acc_10"] = EE_acc_10
                        results["EE_cond_loss_10"] = EE_cond_loss_10
                        results["EE_cond_acc_10"] = EE_cond_acc_10

                        results["num_params"] = num_params
                        results["VC"] = VC
                        results["sum_weight_norms"] = sum_weight_norms
                        results["prod_weight_norms"] = prod_weight_norms

                        torch.save(results, results_p)

                    model_ind += 1


################################
# Model compression
################################

if args.compute_MI_theta_D:
    num_samples = 10
    set_seed(1)

    model_ind = 0
    setting_i = 0
    for lamb_lr in lamb_lrs:
        for arch in archs:
            for decay in decays:
                doing_setting = setting_i in list(range(args.compute_MI_theta_D_only_start, args.compute_MI_theta_D_only_end))

                if doing_setting: swag_models = defaultdict(list)
                for data_instance in data_instances:
                    for seed in seeds:
                        if doing_setting:
                            results_f = os.path.join(args.out_dir, "results_%d.pt" % model_ind)
                            print("loading %s" % results_f)
                            results = torch.load(results_f)
                            swag_models[seed].append(results["swag_model"])

                        model_ind += 1 # increment even if not doing

                if doing_setting:
                    print("%s" % datetime.now())
                    sys.stdout.flush()

                    for seed in seeds:
                        theta_D_key_single_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))

                        MI_theta_D_single_seed = compute_MI_theta_D_single_seed_jensen(swag_models[seed], num_samples, layers=[0, 1])
                        torch.save(MI_theta_D_single_seed, os.path.join(args.out_dir,"%d_theta_D_key_single_%s.pt" % (1, theta_D_key_single_seed)))

                    theta_D_key_multi_seed = clean("%s_%s_%s" % (lamb_lr, arch, decay))
                    MI_theta_D_multi_seed = compute_MI_theta_D_multiseed_jensen(swag_models, num_samples, layers=[0, 1])
                    torch.save(MI_theta_D_multi_seed, os.path.join(args.out_dir, "%d_theta_D_key_multi_%s.pt" % (1, theta_D_key_multi_seed)))

                    print(("MI results for %s (%s): single %s, multi %s" % (setting_i, theta_D_key_multi_seed, MI_theta_D_single_seed, MI_theta_D_multi_seed)))

                setting_i += 1


################################
# Results
################################


use_orig = False # use training set (true) or larger sample of data (false)
suff_base = ""
if not use_orig: suff_base = "_10"
all_gaps = True

if args.results:
    sns.set_style("dark")

    plot = True
    print_summary = False

    vnames_base_invariant = [
        "MI_theta_singles_last",
        "MI_theta_multis_last",
    ]

    vnames_base_extra = [
        "num_params",
        "VC",
        "sum_weight_norms",
        "prod_weight_norms",
    ]

    vnames_entropy = ["EE_loss", "EE_acc", "EE_cond_loss", "EE_cond_acc"]

    vnames_base = ["MI_mcs", "MI_cond_mcs", "MI_jensens", "MI_cond_jensens",
                   "MI_theta_singles", "MI_theta_multis"] + vnames_base_invariant + vnames_base_extra + vnames_entropy

    vnames = vnames_base + [
        "combined_singles_mc", "combined_single_conds_mc",
        "combined_singles_jensen", "combined_single_conds_jensen",

        "combined_singles_mc_noscale", "combined_single_conds_mc_noscale",
        "combined_singles_jensen_noscale", "combined_single_conds_jensen_noscale",

        # others
        "combined_multis_mc", "combined_multi_conds_mc",
        "combined_multis_jensen", "combined_multi_conds_jensen",

        "combined_multis_mc_noscale", "combined_multi_conds_mc_noscale",
        "combined_multis_jensen_noscale", "combined_multi_conds_jensen_noscale",
    ]

    inspect_est_absolutes, inspect_gaps = [], []


    thresh = 0.85
    poly = 2

    for to_vary in ["none"]:
        print("Varying: %s" % to_vary)

        for lamb_lr_i, lamb_lr_curr in enumerate(lamb_lrs):
            details = defaultdict(list)

            if not print_summary: print("\nDoing lamb_lr %s" % lamb_lr_curr)

            GL = defaultdict(list)

            train_losses = defaultdict(list)
            train_accs = defaultdict(list)
            test_losses = defaultdict(list)
            test_accs = defaultdict(list)

            for v_i, vname in enumerate(vnames_base):
                locals()["%s" % vname] = defaultdict(list)

            counted = 0
            skipped1 = 0
            skipped2 = 0
            skipped3 = 0
            model_ind = 0

            metrics = ["tau_GL", "r_GL", "p_GL"]

            keys = {}
            for lamb_lr in lamb_lrs:
                for arch_i, arch in enumerate(archs):
                    for decay_i, decay in enumerate(decays):

                        for data_i, data_instance in enumerate(data_instances):

                            for seed_i, seed in enumerate(seeds):

                                if lamb_lr == lamb_lr_curr:
                                    r = torch.load(os.path.join(args.out_dir, "results_%d.pt" % model_ind))
                                    if (r["train_acc"] > thresh):

                                        if all_gaps or ((not all_gaps) and r["gen_gap_loss"] > 0.):
                                            key = get_key(arch_i, decay_i, data_i, seed_i, to_vary)
                                            keys[key] = 1

                                            GL[key].append(r["gen_gap_loss"])

                                            train_losses[key].append(r["train_loss"])
                                            train_accs[key].append(r["train_acc"])
                                            test_losses[key].append(r["test_loss"])
                                            test_accs[key].append(r["test_acc"])

                                            if not use_orig:
                                                MI_mcs[key].append(to_bits(r["MI_mc" + suff_base]))
                                                MI_cond_mcs[key].append(to_bits(r["MI_cond_mc" + suff_base]))
                                                MI_jensens[key].append(to_bits(r["MI_jensen" + suff_base]))
                                                MI_cond_jensens[key].append(to_bits(r["MI_cond_jensen" + suff_base]))
                                                EE_loss[key].append(to_bits(r["EE_loss" + suff_base]))
                                                EE_acc[key].append(to_bits(r["EE_acc" + suff_base]))
                                                EE_cond_loss[key].append(to_bits(r["EE_cond_loss" + suff_base]))
                                                EE_cond_acc[key].append(to_bits(r["EE_cond_acc" + suff_base]))
                                            else:
                                                MI_mcs[key].append(to_bits(r["diagnostics"]["MI_mc" + suff_base]))
                                                MI_cond_mcs[key].append(to_bits(r["diagnostics"]["MI_cond_mc" + suff_base]))
                                                MI_jensens[key].append(to_bits(r["diagnostics"]["MI_jensen" + suff_base]))
                                                MI_cond_jensens[key].append(to_bits(r["diagnostics"]["MI_cond_jensen" + suff_base]))
                                                EE_loss[key].append(to_bits(r["diagnostics"]["EE_loss" + suff_base]))
                                                EE_acc[key].append(to_bits(r["diagnostics"]["EE_acc" + suff_base]))
                                                EE_cond_loss[key].append(to_bits(r["diagnostics"]["EE_cond_loss" + suff_base]))
                                                EE_cond_acc[key].append(to_bits(r["diagnostics"]["EE_cond_acc" + suff_base]))

                                            theta_D_key_single_seed = clean("%s_%s_%s_%s" % (seed, lamb_lr, arch, decay))
                                            theta_single = to_bits(torch.load(os.path.join(args.out_dir, "%d_theta_D_key_single_%s.pt" % (1, theta_D_key_single_seed))))

                                            theta_D_key_multi_seed = clean("%s_%s_%s" % (lamb_lr, arch, decay))
                                            theta_multi = to_bits(torch.load(os.path.join(args.out_dir, "%d_theta_D_key_multi_%s.pt" % (1, theta_D_key_multi_seed))))

                                            MI_theta_singles[key].append(theta_single[0].item())
                                            MI_theta_multis[key].append(theta_multi[0].item())

                                            MI_theta_singles_last[key].append(theta_single[1].item())
                                            MI_theta_multis_last[key].append(theta_multi[1].item())

                                            num_params[key].append(r["num_params"])
                                            VC[key].append(r["VC"])
                                            sum_weight_norms[key].append(r["sum_weight_norms"])
                                            prod_weight_norms[key].append(r["prod_weight_norms"])

                                            details[key].append(arch_name(arch))
                                        else:
                                            skipped3 += 1

                                    else:
                                        skipped2 += 1

                                    counted += 1

                                model_ind += 1

            print("Counts %s, skipped err %s, thresh %s, gen gap %s" % (counted, skipped1, skipped2, skipped3))

            # print("--- Inspect --- ")
            #
            # for inspect_i in range(len(inspect_est_absolutes)):
            #     print((inspect_est_absolutes[inspect_i], inspect_gaps[inspect_i], inspect_est_absolutes[inspect_i] - inspect_gaps[inspect_i]))
            #
            # print("--- Summary ---")
            #
            # inspect_est_absolutes = torch.tensor(inspect_est_absolutes)
            # inspect_gaps = torch.tensor(inspect_gaps)
            # inspect_diff = (inspect_est_absolutes - inspect_gaps).abs()
            # print((inspect_diff.min(), inspect_diff.max(), inspect_diff.mean(), inspect_diff.std()))
            #
            # print("--------")

            for metric in metrics:
                locals()["results_%s" % metric] = defaultdict(dict)

            for key in keys:
                GL_curr = np.array(GL[key])

                archs_key = details[key]

                train_losses_curr = np.array(train_losses[key])
                train_accs_curr = np.array(train_accs[key])
                test_losses_curr = np.array(test_losses[key])
                test_accs_curr = np.array(test_accs[key])

                MI_mcs_curr = (np.array(MI_mcs[key])) # all models for key
                MI_cond_mcs_curr = (np.array(MI_cond_mcs[key]))

                MI_jensens_curr = (np.array(MI_jensens[key]))
                MI_cond_jensens_curr = (np.array(MI_cond_jensens[key]))

                EE_loss_curr = (np.array(EE_loss[key]))
                EE_acc_curr = (np.array(EE_acc[key]))
                EE_cond_loss_curr = (np.array(EE_cond_loss[key]))
                EE_cond_acc_curr = (np.array(EE_cond_acc[key]))

                MI_theta_singles_curr = (np.array(MI_theta_singles[key])) # each one combined with the above 4
                MI_theta_multis_curr = (np.array(MI_theta_multis[key]))

                MI_theta_singles_last_curr = (np.array(MI_theta_singles_last[key]))  # each one combined with the above 4
                MI_theta_multis_last_curr = (np.array(MI_theta_multis_last[key]))

                num_params_curr = np.array(num_params[key])
                VC_curr = np.array(VC[key])
                sum_weight_norms_curr = np.array(sum_weight_norms[key])
                prod_weight_norms_curr = np.array(prod_weight_norms[key])

                scales = [[], []]
                for vname in ["MI_mcs_curr", "MI_cond_mcs_curr", "MI_jensens_curr", "MI_cond_jensens_curr"]:
                    v = locals()[vname]

                    scales[0].append(v.mean() / MI_theta_singles_curr.mean())
                    scales[1].append(v.mean() / MI_theta_multis_curr.mean())

                print("Scales for %s %s" % (lamb_lr_curr, lamb_lr_i))
                print(scales)

                combined_singles_mc_curr = scales[0][0] * MI_theta_singles_curr + MI_mcs_curr
                combined_single_conds_mc_curr = scales[0][1] * MI_theta_singles_curr + MI_cond_mcs_curr

                combined_singles_jensen_curr = scales[0][2] * MI_theta_singles_curr + MI_jensens_curr
                combined_single_conds_jensen_curr = scales[0][3] * MI_theta_singles_curr + MI_cond_jensens_curr

                combined_multis_mc_curr = scales[1][0] * MI_theta_multis_curr + MI_mcs_curr
                combined_multi_conds_mc_curr = scales[1][1] * MI_theta_multis_curr + MI_cond_mcs_curr

                combined_multis_jensen_curr = scales[1][2] * MI_theta_multis_curr + MI_jensens_curr
                combined_multi_conds_jensen_curr = scales[1][3] * MI_theta_multis_curr + MI_cond_jensens_curr

                for ii in range(combined_single_conds_mc_curr.shape[0]):
                    if abs(combined_single_conds_mc_curr[ii] - combined_single_conds_mc_curr.mean()) / combined_single_conds_mc_curr.std() >= 2:
                        print(("outlier", combined_single_conds_mc_curr[ii], details[key][ii]))

                # noscale

                combined_singles_jensen_noscale_curr = MI_theta_singles_curr + MI_jensens_curr
                combined_single_conds_jensen_noscale_curr = MI_theta_singles_curr + MI_cond_jensens_curr

                combined_multis_jensen_noscale_curr = MI_theta_multis_curr + MI_jensens_curr
                combined_multi_conds_jensen_noscale_curr = MI_theta_multis_curr + MI_cond_jensens_curr

                combined_singles_mc_noscale_curr = MI_theta_singles_curr + MI_mcs_curr
                combined_single_conds_mc_noscale_curr = MI_theta_singles_curr + MI_cond_mcs_curr

                combined_multis_mc_noscale_curr = MI_theta_multis_curr + MI_mcs_curr
                combined_multi_conds_mc_noscale_curr = MI_theta_multis_curr + MI_cond_mcs_curr

                if plot:
                    fig, axarr = plt.subplots(len(vnames), 2, figsize=(2 * 3, len(vnames) * 3)) # all
                    fig2, axarr2 = plt.subplots(1, figsize=(5, 5))
                    fig3, axarr3 = plt.subplots(1, figsize=(5, 5))
                    fig4, axarr4 = plt.subplots(1, figsize=(5, 5))
                    fig5, axarr5 = plt.subplots(1, figsize=(5, 5))
                    fig6, axarr6 = plt.subplots(1, figsize=(5, 5))
                    fig7, axarr7 = plt.subplots(1, figsize=(5, 5))

                for v_i, vname in enumerate(vnames):
                    v = locals()["%s_curr" % vname]

                    for metric in metrics: # wht
                        locals()["%s_curr" % metric] = np.nan

                    try:
                        tau_GL_curr, _ = stats.kendalltau(GL_curr, v)
                        r_GL_curr, _ = stats.spearmanr(GL_curr, v)
                        p_GL_curr, _ = stats.pearsonr(GL_curr, v)
                    except Exception as e:
                        print(e)
                        continue

                    for metric in metrics:
                        locals()["results_%s" % metric][vname][key] = locals()["%s_curr" % metric]

                    if not print_summary:
                        pretty_name = vname
                        if vname in vnames_pretty: pretty_name = vnames_pretty[vname]
                        print("%s \t& %.4f & %.4f & %.4f \\\\" % (pretty_name, r_GL_curr, p_GL_curr, tau_GL_curr))

                    if plot:
                        # all
                        axarr[v_i, 0].scatter(GL_curr, v)

                        z0 = np.polyfit(GL_curr, v, poly)
                        x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                        axarr[v_i, 0].plot(x, np.polyval(z0, x), "r--")

                        axarr[v_i, 0].set_ylabel(vname)

                        axarr[v_i, 0].set_xlabel("GL")

                        axarr[v_i, 0].set_title("tau %.3f, r %.3f, \n p %s" % (tau_GL_curr, r_GL_curr, p_GL_curr))

                        # individuals
                        if vname == "MI_cond_jensens":
                            # axarr2.scatter(GL_curr, v)

                            df1 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df1, x="GL_curr", y="v", hue="arch", palette="deep", s=60, ax=axarr2)

                            z2 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z2, x)
                            assert res.shape == x.shape

                            #axarr2.plot(x, res, "r--")
                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr2, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr2.legend(title="Width", loc="lower right")
                            else:
                                axarr2.legend(title="Width", loc="upper left")

                            axarr2.set_ylabel(vnames_pretty[vname])
                            axarr2.set_xlabel("Generalization gap in loss")
                            axarr2.set_title("Pearson correlation: %.3f" % (p_GL_curr))
                            #axarr2.set_xticklabels(axarr2.get_xticklabels(), rotation=45)

                        if vname == "combined_singles_jensen":
                            # axarr2.scatter(GL_curr, v)

                            df1 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df1, x="GL_curr", y="v", hue="arch", palette="deep", s=60, ax=axarr3)

                            z2 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z2, x)
                            assert res.shape == x.shape

                            #axarr2.plot(x, res, "r--")
                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr3, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr3.legend(title="Width", loc="lower right")
                            else:
                                axarr3.legend(title="Width", loc="upper left")

                            axarr3.set_ylabel(vnames_pretty[vname])
                            axarr3.set_xlabel("Generalization gap in loss")
                            axarr3.set_title("Pearson correlation: %.3f" % (p_GL_curr))
                            #axarr2.set_xticklabels(axarr2.get_xticklabels(), rotation=45)

                        if vname == "combined_single_conds_jensen":
                            #axarr3.scatter(GL_curr, v)
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr4)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr4, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr4.legend(title="Width", loc="lower right")
                            else:
                                axarr4.legend(title="Width", loc="upper left")

                            axarr4.set_ylabel(vnames_pretty[vname])
                            axarr4.set_xlabel("Generalization gap in loss")
                            axarr4.set_title("Pearson correlation: %.3f" % (p_GL_curr))

                        if vname == "combined_single_conds_mc":
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr5)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr5, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr5.legend(title="Width", loc="lower right")
                            else:
                                axarr5.legend(title="Width", loc="upper left")

                            axarr5.set_ylabel(vnames_pretty[vname])
                            axarr5.set_xlabel("Generalization gap in loss")
                            axarr5.set_title("Pearson correlation: %.3f" % (p_GL_curr))

                        if vname == "EE_loss":
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr6)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr6, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr6.legend(title="Width", loc="lower right")
                            else:
                                axarr6.legend(title="Width", loc="upper left")

                            axarr6.set_ylabel(vnames_pretty[vname])
                            axarr6.set_xlabel("Generalization gap in loss")
                            axarr6.set_title("Pearson correlation: %.3f" % (p_GL_curr))

                        if vname == "EE_cond_loss":
                            df2 = create_dataframe(GL_curr, v, archs_key)
                            sns.scatterplot(data=df2, x="GL_curr", y="v",
                                            hue="arch", palette="deep", s=60, ax=axarr7)

                            z3 = np.polyfit(GL_curr, v, poly)
                            x = np.linspace(GL_curr.min(), GL_curr.max(), 30)
                            res = np.polyval(z3, x)
                            assert res.shape == x.shape

                            sns.lineplot(x=x, y=res, palette="deep", ax=axarr7, linestyle="--")

                            if lamb_lr_i == 0:
                                axarr7.legend(title="Width", loc="lower right")
                            else:
                                axarr7.legend(title="Width", loc="upper left")

                            axarr7.set_ylabel(vnames_pretty[vname])
                            axarr7.set_xlabel("Generalization gap in loss")
                            axarr7.set_title("Pearson correlation: %.3f" % (p_GL_curr))

                if plot:
                    plt.tight_layout()
                    fig.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig2.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_MI_cond_jensens.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig3.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_combined_singles_jensen.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig4.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_combined_single_conds_jensen.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig5.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_combined_single_conds_mc.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig6.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_EE_loss.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")
                    fig7.savefig(os.path.join(args.out_dir, "summary_lr_%s_key_%s_EE_cond_loss.pdf" % (lamb_lr_curr, key)), bbox_inches="tight")

                    plt.close("all")

            for metric in metrics:
                locals()["repr_best_%s" % metric] = (-np.inf, None)
                locals()["model_incl_best_%s" % metric] = (-np.inf, None)

            for v_i, vname in enumerate(vnames):
                if vname in ["MI_mcs", "MI_cond_mcs", "MI_jensens", "MI_cond_jensens"]:
                    pref = "repr"
                else:
                    pref = "model_incl"

                for metric in metrics:
                    avg_results = np.array(list(locals()["results_%s" % metric][vname].values())).mean()
                    if avg_results > locals()["%s_best_%s" % (pref, metric)][0]:
                        locals()["%s_best_%s" % (pref, metric)] = (avg_results, vname)

            if print_summary:
                print("Results for %s, %s" % (lamb_lr_curr, to_vary))
                for metric in metrics:
                    print("%s: %s" % ("repr_best_%s" % metric, locals()["repr_best_%s" % metric]))
                    print("%s: %s" % ("model_incl_best_%s" % metric, locals()["model_incl_best_%s" % metric]))
