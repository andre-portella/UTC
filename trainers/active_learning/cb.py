from .AL import AL
import torch
import numpy as np

from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from scipy.spatial.distance import cdist

from .sample_selection import Selector
from .turtle_data import prepare_turtle_data
from .turtle_runner import run_turtle
from dassl.data import DataManager
from .umap_representations import save_data_for_umap 


class CB(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, val_set, n_class, device, idx, dataset, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.labeled_in_set = val_set
        self.device = device
        self.idx = idx
        self.dataset = dataset


    def get_features(self):
        if self.idx:
            self.model.eval()
        labeled_features, unlabeled_features = None, None

        with torch.no_grad():
            dm = DataManager(self.cfg)
            val_loader = dm.val_loader
            test_loader = dm.test_loader

            
            labeled_in_loader = build_data_loader(
                self.cfg,
                data_source=self.labeled_in_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            labels_org, labels_pse = np.array([]), np.array([])
            scores = np.array([])

            for data in labeled_in_loader:
                inputs = data["img"].cuda()
                labels = data["label"].cuda()
                preds, img_features, txt_features = self.model(inputs, get_feature=True, get_text_feature=True)
                p_labels = torch.argmax(preds, dim=1)

                labels_org = np.append(labels_org, labels.cpu().numpy())
                labels_pse = np.append(labels_pse, p_labels.cpu().numpy())

                preds = torch.nn.functional.softmax(preds, dim=1)
                score = np.max(preds.cpu().numpy(), axis=1)
                scores = np.append(scores, score)

                dot_features = torch.matmul(preds, txt_features)
                features = torch.cat([img_features, dot_features], axis=1)
                labeled_features = features if labeled_features is None else torch.cat((labeled_features, features), 0)

            for data in unlabeled_loader:
                inputs = data["img"].cuda()
                labels = data["label"]
                if self.idx:
                    preds, img_features, txt_features = self.model(inputs, get_feature=True, get_text_feature=True)
                else:
                    preds, img_features, txt_features = self.model.model_inference(inputs, get_feature=True)
                p_labels = torch.argmax(preds, dim=1)

                labels_org = np.append(labels_org, labels.cpu().numpy())
                labels_pse = np.append(labels_pse, p_labels.cpu().numpy())

                preds = torch.nn.functional.softmax(preds, dim=1)
                score = np.max(preds.cpu().numpy(), axis=1)
                scores = np.append(scores, score)

                dot_features = torch.matmul(preds, txt_features)
                features = torch.cat([img_features, dot_features], axis=1)
                unlabeled_features = features if unlabeled_features is None else torch.cat((unlabeled_features, features), 0)



            # generate validation set features
            val_features, val_pse, val_org = None, None, None
            if hasattr(self.dataset, "val") and self.dataset.val is not None:
                v_org, v_pse = np.array([]), np.array([])
                for data in val_loader:
                    inputs, labels = data["img"].cuda(), data["label"]
                    preds, img_features, txt_features = self.model.model_inference(inputs, get_feature=True)
                    p_labels = torch.argmax(preds, dim=1)

                    v_org = np.append(v_org, labels.cpu().numpy())
                    v_pse = np.append(v_pse, p_labels.cpu().numpy())

                    preds = torch.nn.functional.softmax(preds, dim=1)
                    dot_features = torch.matmul(preds, txt_features)
                    features = torch.cat([img_features, dot_features], dim=1)
                    val_features = features if val_features is None else torch.cat((val_features, features), 0)
                val_org, val_pse = v_org, v_pse

            # generate test set features
            test_features, test_pse, test_org = None, None, None
            if hasattr(self.dataset, "test") and self.dataset.test is not None:
                t_org, t_pse = np.array([]), np.array([])
                for data in test_loader:
                    inputs, labels = data["img"].cuda(), data["label"]
                    preds, img_features, txt_features = self.model.model_inference(inputs, get_feature=True)
                    p_labels = torch.argmax(preds, dim=1)

                    t_org = np.append(t_org, labels.cpu().numpy())
                    t_pse = np.append(t_pse, p_labels.cpu().numpy())

                    preds = torch.nn.functional.softmax(preds, dim=1)
                    dot_features = torch.matmul(preds, txt_features)
                    features = torch.cat([img_features, dot_features], dim=1)
                    test_features = features if test_features is None else torch.cat((test_features, features), 0)
                test_org, test_pse = t_org, t_pse

        return labeled_features, unlabeled_features, scores, labels_org, labels_pse, val_features, val_pse, val_org, test_features, test_pse, test_org



    def turtle_uncert(self, labeled, n_query, strategy):
        if self.idx != 0:
            label_len = labeled.size()[0]
            data = list(self.labeled_in_set) + list(self.unlabeled_set)
            num_clusters = self.n_class + label_len
        else:
            label_len = 0
            data = list(self.unlabeled_set)
            num_clusters = self.n_class


        train_features, train_labels = prepare_turtle_data(
            model=self.model,
            cfg=self.cfg,
            unlabeled_dst=data,
            dataset=self.dataset,
            round=self.idx
        )

        if isinstance(train_features, torch.Tensor):
            train_features = train_features.numpy()

        dim = train_features.shape[1]
        task_phi, cluster_acc_train, cluster_acc_val, preds_train, preds_test = run_turtle(self, num_clusters, dim, self.idx)

        with torch.no_grad():
            inputs = torch.from_numpy(train_features).to(self.device).to(torch.float32)
            #inputs_norm = inputs / torch.norm(inputs, dim=1, keepdim=True)
            #logits_norm = task_phi(inputs_norm)
            logits = task_phi(inputs)
            probs = torch.softmax(logits, dim=1)
            pseudolabels = torch.argmax(probs, dim=1).cpu().numpy()

        uniques = np.unique(pseudolabels)
        selector = Selector(task_phi=task_phi, device=self.device, U_index=self.U_index)

        if self.idx == 0:
            for i, c in enumerate(uniques):
                c_indices = np.where(pseudolabels == c)[0]
                features_cluster = torch.from_numpy(train_features[c_indices]).float()
                probs_cluster = probs[c_indices]
                Q_cluster_cand, _ = selector.run(strategy=strategy, n_query=1, features_cluster=features_cluster, probs_cluster=probs_cluster)
                selection = torch.tensor([torch.tensor(c_indices[int(Q_cluster_cand[0])])])
                final_indices = selection if i == 0 else torch.cat((final_indices, selection), 0)
        else:
            ratio_per_clusters, size_per_clusters = {}, {}
            num_per_clusters = {c: 0 for c in uniques}
            for c in uniques:
                c_indices = np.where(pseudolabels == c)[0]
                size_per_clusters[c] = len(c_indices)
                for idx in c_indices:
                    if idx < label_len:
                        num_per_clusters[c] += 1
                ratio_per_clusters[c] = num_per_clusters[c] / size_per_clusters[c]

            budget_per_clusters = {c: 0 for c in uniques}
            for i in range(n_query):
                min_value = min(ratio_per_clusters.values())
                min_keys = [key for key, value in ratio_per_clusters.items() if value == min_value]
                max_key = max(min_keys, key=size_per_clusters.get)
                budget_per_clusters[max_key] += 1
                num_per_clusters[max_key] += 1
                ratio_per_clusters[max_key] = num_per_clusters[max_key] / size_per_clusters[max_key]

            selects = []
            for c in uniques:
                if budget_per_clusters[c] == 0:
                    continue

                c_indices = np.where(pseudolabels == c)[0]
                unlabeled_in_cluster = [idx for idx in c_indices if idx >= label_len]
                if len(unlabeled_in_cluster) == 0:
                    continue

                assert budget_per_clusters[c] == 1

                cluster_features = torch.from_numpy(train_features[unlabeled_in_cluster]).float()
                probs_cluster = probs[unlabeled_in_cluster]
                Q_cluster_cand, _ = selector.run(strategy=strategy, n_query=1, features=cluster_features, probs_cluster=probs_cluster)

                for local_rank in Q_cluster_cand:
                    idx = unlabeled_in_cluster[int(local_rank)]
                    if idx in selects:
                        continue
                    selection = torch.tensor([torch.tensor(idx - label_len)])
                    final_indices = selection if len(selects) == 0 else torch.cat((final_indices, selection), 0)
                    selects.append(idx)
                    break

        return final_indices.cpu().numpy(), cluster_acc_train, cluster_acc_val, preds_train, preds_test

    def select(self, n_query, **kwargs):
        labeled_features, unlabeled_features, scores, labels_org, labels_pse, val_features, val_pse, val_org, test_features, test_pse, test_org = self.get_features()
        selected_indices, cluster_acc_train, cluster_acc_val, preds_train, preds_test = self.turtle_uncert(labeled_features, n_query, self.cfg.STRATEGY)
        Q_index = [self.U_index[int(idx)] for idx in selected_indices]


        # labeled + unlabeled
        train_features_base = unlabeled_features

        save_data_for_umap(
            self.cfg,
            self.idx,
            self.labeled_in_set,

            train_features=train_features_base,
            model_train_pse=labels_pse,
            turtle_train_pse=preds_train,
            labels_org=labels_org,

            val_features=val_features,
            val_model_pse=val_pse,
            val_turtle_pse=None,
            val_org=val_org,

            test_features=test_features,
            test_model_pse=test_pse,
            test_turtle_pse=preds_test,
            test_org=test_org,
        )

        return Q_index, cluster_acc_train, cluster_acc_val