import os
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from scipy.io import loadmat
from tqdm import tqdm
from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN

def dump_metric_inputs(run_tag: str, y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray):
    os.makedirs("artifacts", exist_ok=True)

    # Save the exact arrays that feed AP/AUC/F1
    path = f"artifacts/metric_inputs_{run_tag}.npz"
    np.savez_compressed(path, y_true=y_true, y_score=y_score, y_pred=y_pred)

    # Confusion components (drives F1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    # Also write a human-readable summary
    txt_path = f"artifacts/metric_inputs_{run_tag}.txt"
    with open(txt_path, "w") as f:
        f.write(f"n={len(y_true)} pos={int((y_true==1).sum())} neg={int((y_true==0).sum())}\n")
        f.write(f"TP={tp} FP={fp} TN={tn} FN={fn}\n")
        f.write("First 20 rows (y_true, y_score, y_pred):\n")
        for i in range(min(20, len(y_true))):
            f.write(f"{i}: {int(y_true[i])}, {float(y_score[i]):.6f}, {int(y_pred[i])}\n")

    print(f"[metric dump] wrote: {path}")
    print(f"[metric dump] wrote: {txt_path}")

def log_epoch_metrics(experiment, fold: int, epoch: int, metrics: dict):
    if not experiment:
        return
    # fold is 1-based in logs for readability
    metrics = dict(metrics)
    metrics["fold"] = int(fold)
    metrics["epoch"] = int(epoch)
    experiment.log_metrics(metrics, step=epoch)

def log_test_inputs_and_counts(experiment, y_true, y_score, y_pred, run_tag: str):
    if not experiment:
        return

    os.makedirs("artifacts", exist_ok=True)

    # Save raw arrays used by roc_auc_score / average_precision_score / f1_score
    arr_path = f"artifacts/test_inputs_{run_tag}.npz"
    np.savez_compressed(
        arr_path,
        y_true=y_true.astype(np.int64),
        y_score=y_score.astype(np.float32),
        y_pred=y_pred.astype(np.int64),
    )
    experiment.log_asset(arr_path)

    # Confusion counts (these are the “numbers” that drive F1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    experiment.log_metrics({
        "test/tp": tp,
        "test/fp": fp,
        "test/tn": tn,
        "test/fn": fn,
        "test/n": int(len(y_true)),
        "test/pos_rate": float((y_true == 1).mean()),
    })

    # Curves used for AP (PR curve) and AUC (ROC curve)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)

    curves_path = f"artifacts/test_curves_{run_tag}.npz"
    np.savez_compressed(
        curves_path,
        precision=prec, recall=rec, pr_thresholds=pr_thresh,
        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresh
    )
    experiment.log_asset(curves_path)
    # experiment.log_asset(curves_path, asset_type="curves")

def rgtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame, nei_att_head, experiment=None):
    
    global_step = 0   # <---- put it here
    
    # torch.autograd.set_detect_anomaly(True)
    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}

    neigh_padding_dict = {}
    nei_feat = []
    if isinstance(neigh_features, pd.DataFrame):  # otherwise []
        # if null it is []
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
        
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
            device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = NodeDataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = NodeDataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        model = RGTAN(in_feats=feat_df.shape[1],
                      hidden_dim=args['hid_dim']//4,
                      n_classes=2,
                      heads=[4]*args['n_layers'],
                      activation=nn.PReLU(),
                      n_layers=args['n_layers'],
                      drop=args['dropout'],
                      device=device,
                      gated=args['gated'],
                      ref_df=feat_df,
                      cat_features=cat_feat,
                      neigh_features=nei_feat,
                      nei_att_head=nei_att_head).to(device)
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
                                   4000, 12000], gamma=0.3)

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)
        start_epoch, max_epochs = 0, 2000
        for epoch in range(start_epoch, args['max_epochs']):
            train_loss_list = []

            train_probs_all = []
            train_labels_all = []

            # train_acc_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # print(f"loading batch data...")
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                       seeds, input_nodes, device, blocks)
                # print(f"load {step}")

                # batch_neighstat_inputs: {"degree":(|batch|, degree_dim)}

                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0

                train_loss = loss_fn(train_batch_logits, batch_labels)
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                score = torch.softmax(train_batch_logits.detach(), dim=1)[:, 1].cpu().numpy()
                train_probs_all.append(score)
                train_labels_all.append(batch_labels.detach().cpu().numpy())

                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                    ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()
                    

                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(), score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()

            val_probs_all = []
            val_labels_all = []

            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                           seeds, input_nodes, device, blocks)

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                    
                    # score = torch.softmax(val_batch_logits.detach(), dim=1)[:, 1].cpu().numpy()
                    # val_probs_all.append(score)
                    # val_labels_all.append(batch_labels.detach().cpu().numpy())

                    # oof_predictions[seeds] = val_batch_logits
                    # mask = batch_labels == 2
                    # val_batch_logits = val_batch_logits[~mask]
                    # batch_labels = batch_labels[~mask]
                    
                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]

                    score = torch.softmax(val_batch_logits.detach(), dim=1)[:, 1].cpu().numpy()
                    val_probs_all.append(score)
                    val_labels_all.append(batch_labels.detach().cpu().numpy())

                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + \
                        loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(
                        val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * \
                        torch.tensor(
                            batch_labels.shape[0])  # how many in this batch is right!
                    val_all_list = val_all_list + \
                        batch_labels.shape[0]  # how many val nodes
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()
                        

                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          step,
                                                                          val_loss_list/val_all_list,
                                                                          average_precision_score(
                                                                              batch_labels.cpu().numpy(), score),
                                                                          val_batch_pred.detach(),
                                                                          roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except:
                            pass

            # val_acc_list/val_all_list, model)
            earlystoper.earlystop(val_loss_list/val_all_list, model)

            # ---- Epoch-level Comet logging (RGTAN) ----
            if experiment:
                # ----- TRAIN METRICS -----
                if len(train_probs_all) > 0:
                    tr_scores = np.concatenate(train_probs_all)
                    tr_true = np.concatenate(train_labels_all)

                    try:
                        tr_auc = roc_auc_score(tr_true, tr_scores)
                    except Exception:
                        tr_auc = float("nan")

                    try:
                        tr_ap = average_precision_score(tr_true, tr_scores)
                    except Exception:
                        tr_ap = float("nan")
                else:
                    tr_auc, tr_ap = float("nan"), float("nan")

                # ----- VALIDATION METRICS -----
                if len(val_probs_all) > 0:
                    va_scores = np.concatenate(val_probs_all)
                    va_true = np.concatenate(val_labels_all)

                    try:
                        va_auc = roc_auc_score(va_true, va_scores)
                    except Exception:
                        va_auc = float("nan")

                    try:
                        va_ap = average_precision_score(va_true, va_scores)
                    except Exception:
                        va_ap = float("nan")
                else:
                    va_auc, va_ap = float("nan"), float("nan")

                # ----- LOG TO COMET -----
                log_epoch_metrics(
                    experiment,
                    fold=fold + 1,
                    epoch=epoch,
                    metrics={
                        "train/loss_mean": float(np.mean(train_loss_list)) if len(train_loss_list) else float("nan"),
                        "train/ap_epoch": float(tr_ap),
                        "train/auc_epoch": float(tr_auc),
                        "val/loss_epoch": float((val_loss_list / val_all_list).detach().cpu().item()),
                        "val/ap_epoch": float(va_ap),
                        "val/auc_epoch": float(va_auc),
                        "val/earlystop_best": float(earlystoper.best_cv),
                    }
                )
            # ---- End epoch-level logging ----

            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = NodeDataLoader(graph,
                                         test_ind,
                                         test_sampler,
                                         use_ddp=False,
                                         device=device,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=0,
                                         )
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # print(input_nodes)
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                       seeds, input_nodes, device, blocks)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(torch.argmax(
                    test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    # mask = y_target == 2
    # y_target[mask] = 0
    # my_ap = average_precision_score(y_target, torch.softmax(
    #     oof_predictions, dim=1).cpu()[train_idx, 1])
    
    y_oof = labels[train_idx].cpu().numpy()
    y_oof = np.where(y_oof == 2, 0, y_oof)

    my_ap = average_precision_score(
        y_oof,
        torch.softmax(oof_predictions, dim=1).cpu().numpy()[train_idx, 1]
)

    print("NN out of fold AP is:", my_ap)

    b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
        'cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    run_tag = f"{args['method']}_{args['dataset']}"
    dump_metric_inputs(run_tag, y_target, test_score, test_score1)

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))

    if experiment:
        run_tag = f"{args['method']}_{args['dataset']}"
        log_test_inputs_and_counts(experiment, y_target, test_score, test_score1, run_tag)

def loda_rgtan_data(dataset: str, test_size: float):
    # prefix = "./antifraud/data/"
    prefix = "data/"
    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]

        
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        #####
        neigh_features = []
        #####
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        #######
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        #######

        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        index = list(range(len(labels)))

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.6,
                                                                random_state=2, shuffle=True)
        feat_neigh = pd.read_csv(
            prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(
                prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        try:
            feat_neigh = pd.read_csv(
                prefix + "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except:
            print("no neighbohood feature used.")

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
