import os

# IMPORTANT: must be imported before sklearn/torch/etc to enable auto-logging cleanly
from comet_ml import Experiment

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
import dgl
from scipy.io import loadmat
import yaml

logger = logging.getLogger(__name__)
# sys.path.append("..")


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']  # dict

    # if method in ['']:
    #     yaml_file = "config/base_cfg.yaml"
    if method in ['mcnn']:
        yaml_file = "config/mcnn_cfg.yaml"
    elif method in ['stan']:
        yaml_file = "config/stan_cfg.yaml"
    elif method in ['stan_2d']:
        yaml_file = "config/stan_2d_cfg.yaml"
    elif method in ['stagn']:
        yaml_file = "config/stagn_cfg.yaml"
    elif method in ['gtan']:
        yaml_file = "config/gtan_cfg.yaml"
    elif method in ['rgtan']:
        yaml_file = "config/rgtan_cfg.yaml"
    elif method in ['hogrl']:
        yaml_file = "config/hogrl_cfg.yaml"
        
    else:
        raise NotImplementedError("Unsupported method.")

    # config = Config().get_config()
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    return args


def base_load_data(args: dict):
    # load S-FFSD dataset for base models
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    # for ICONIP16 & AAAI20
    if args['method'] == 'stan':
        if os.path.exists("data/tel_3d.npy"):
            return
        features, labels = span_data_3d(feat_df)
    else:
        if os.path.exists("data/tel_2d.npy"):
            return
        features, labels = span_data_2d(feat_df)
    num_trans = len(feat_df)
    trf, tef, trl, tel = train_test_split(
        features, labels, train_size=train_size, stratify=labels, shuffle=True)
    trf_file, tef_file, trl_file, tel_file = args['trainfeature'], args[
        'testfeature'], args['trainlabel'], args['testlabel']
    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return

def start_comet_experiment(args: dict):
    # If COMET_API_KEY not set, just run without Comet (no crashes)
    if not os.getenv("COMET_API_KEY"):
        return None

    exp = Experiment(
        project_name=os.getenv("COMET_PROJECT_NAME", "ece-thesis-fraud"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_param_logging=False,   # you control what is logged
        auto_metric_logging=False,
        log_code=True,
    )

    exp.set_name(f"{args.get('method')}-{args.get('dataset')}")
    exp.add_tags([args.get("method", "unknown"), args.get("dataset", "unknown")])

    # Log all YAML args as parameters
    exp.log_parameters(args)

    # Useful extra context
    exp.log_parameter("device", args.get("device"))
    exp.log_parameter("n_fold", args.get("n_fold"))
    exp.log_parameter("max_epochs", args.get("max_epochs"))
    return exp

def main(args):

    experiment = start_comet_experiment(args)

    if args['method'] == 'mcnn':
        from methods.mcnn.mcnn_main import mcnn_main
        base_load_data(args)
        mcnn_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            lr=args['lr'],
            device=args['device']
        )
    elif args['method'] == 'stan_2d':
        from methods.stan.stan_2d_main import stan_main
        base_load_data(args)
        stan_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            mode='2d',
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )
    elif args['method'] == 'stan':
        from methods.stan.stan_main import stan_main
        base_load_data(args)
        stan_main(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            mode='3d',
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )

    elif args['method'] == 'stagn':
        from methods.stagn.stagn_main import stagn_main, load_stagn_data
        features, labels, g = load_stagn_data(args)
        stagn_main(
            features,
            labels,
            args['test_size'],
            g,
            mode='2d',
            epochs=args['epochs'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )
    elif args['method'] == 'gtan':
        from methods.gtan.gtan_main import gtan_main, load_gtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
            args['dataset'], args['test_size'])
        gtan_main(
            feat_data, g, train_idx, test_idx, labels, args, cat_features, experiment=experiment)
    elif args['method'] == 'rgtan':
        from methods.rgtan.rgtan_main import rgtan_main, loda_rgtan_data
        feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features = loda_rgtan_data(
            args['dataset'], args['test_size'])
        rgtan_main(feat_data, g, train_idx, test_idx, labels, args,
                   cat_features, neigh_features, nei_att_head=args['nei_att_heads'][args['dataset']], experiment=experiment)
    elif args['method'] == 'hogrl':
        from methods.hogrl.hogrl_main import hogrl_main
        hogrl_main(args)
    else:
        raise NotImplementedError("Unsupported method. ")


if __name__ == "__main__":
    main(parse_args())
