import os.path as osp
from random import sample 
import time 
import json 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .active_learning.pcb import PCB
from .active_learning.badge import BADGE
from .active_learning.coreset import Coreset
from .active_learning.entropy import Entropy
from .active_learning.cb import CB
from .active_learning.cbsq import CBSQ

import copy
import os
import csv

from dassl.utils import set_random_seed

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "HAM10000": "a photo of a skin lesion of type {}.", # 
    "KaoKore": "a Japanese artwork of a {}.",

    # INCLUDED
    "CIFAR10_Custom": "a photo of a {}.",
    "STL10_Custom": "a photo of a {}."
}

@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image, get_feature=False):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        if get_feature:
            return logits, image_features, self.text_features
        else:
            return logits


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if not ctx_init.endswith(".json"):
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        classnames = [name.replace("_", " ") for name in classnames]
        n_desc_per_cls = None
        if cfg.TRAINER.COOPAL.ASPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.ASPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        elif cfg.TRAINER.COOPAL.AEPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.AEPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        else:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
       
       
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = embedding.size(0)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
       
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, desc_file=None):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_class_desc=[]
        self.n_cls = len(classnames)
        self.cfg = cfg
        
        if desc_file is not None:
            with open(f"descriptors/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
            classnames = [name.replace("_", " ") for name in classnames]
            for name in classnames:
                name = name.lower()
                self.n_class_desc.append(len(desc_dict[name]))
            
        
    def forward(self, image, get_feature=False, get_text_feature=False):
        image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
    
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        if self.cfg.TRAINER.COOPAL.AEPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(text_features[start:start+n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.cfg.TRAINER.COOPAL.ASPATH:
            tmp = [] 
            start = 0
            for n in self.n_class_desc:
                tmp.append(torch.sum(logits[:, start:start+n], dim=1)/n)
                start += n
            logits = torch.stack(tmp, dim=1)

            tmp = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(text_features[start:start+n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)
        
        if get_feature:
            if get_text_feature:
                return logits, image_features, text_features
            else:
                return logits, image_features
        else:
            return logits

    # ADDED
    def model_inference(self, image, get_feature=False):
        return self.forward(image, get_feature=get_feature, get_text_feature=True)
        
        


@TRAINER_REGISTRY.register()
class ALVLM(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acc = []
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.TRAINER.COOPAL.ASPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.ASPATH)
        elif cfg.TRAINER.COOPAL.AEPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.AEPATH)
        else:
            self.model = CustomCLIP(cfg, classnames, clip_model)
        print(self.model)
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(f"prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            print(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def before_train(self):
        print("INITIALIZE the prompts weights")
        self.build_model()
        
    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.acc.append(self.test())
            
        # Close writer
        self.close_writer()
        
    def train(self):
        """Generic training loop executing a single active learning strategy."""
        dataset = build_dataset(self.cfg)
        print(f"dataset length: {len(dataset.train_x)}")

        rounds = 9

        unlabeled_dst = dataset.train_x
        n_query = dataset.get_num_classes(unlabeled_dst)
        n_cand = int(len(unlabeled_dst) * self.cfg.TRAINER.COOPAL.GAMMA)
        
        strategy = self.cfg.STRATEGY.lower()

        U_index = list(range(len(unlabeled_dst)))
        strategy_train_x = []
        budgets = [(i + 1) * n_query for i in range(rounds)]
        round_data = []

        dataset._train_x = []

        for i in range(rounds):
            print(f"\n================ ROUND {i} ================\n")
            start = time.time()
            val_x = dataset._train_x.copy()

            cluster_acc_train, cluster_acc_val = 0.0, 0.0
            budget_saving, corr_ratio = 0, 0.0
            len_selected_indices, len_q_index = n_query, 0


            if self.cfg.TRAINER.COOPAL.METHOD== "cb":
                if i == 0:
                    zsclip = ZeroshotCLIP(self.cfg)
                    zsclip.build_model()
                    selector = CB(self.cfg, zsclip, unlabeled_dst, U_index, val_x, n_query, self.device, i)
                else:
                    selector = CB(self.cfg, self.model, unlabeled_dst, U_index, val_x, n_query, self.device, i)
                idx = selector.select(n_query)

            elif self.cfg.TRAINER.COOPAL.METHOD == "cbsq":
                if i == 0:
                    zsclip = ZeroshotCLIP(self.cfg)
                    zsclip.build_model()
                    selector = CB(self.cfg, zsclip, unlabeled_dst, U_index, val_x, n_query, self.device, i, dataset)
                    idx, cluster_acc_train, cluster_acc_val = selector.select(n_query)
                    len_selected_indices = len(idx)
                else:
                    selector = CBSQ(self.cfg, self.model, unlabeled_dst, U_index, strategy_train_x, n_query, self.device, i, dataset)
                    idx, budget_saving, corr_ratio, cluster_acc_train, cluster_acc_val = selector.select(n_query)

                    budgets[i] -= budget_saving
                    print(f">>> Budget saving: {budget_saving} | Correct pseudo-label ratio: {corr_ratio:.4f}")


            elif self.cfg.TRAINER.COOPAL.METHOD == "random" or i == 0:
                idx = sample(U_index, n_query)

            elif self.cfg.TRAINER.COOPAL.METHOD == "entropy":
                selector = Entropy(self.cfg, self.model, unlabeled_dst, U_index, n_query, self.device)
                idx = selector.select(n_query)

            elif self.cfg.TRAINER.COOPAL.METHOD == "badge":
                selector = BADGE(self.cfg, self.model, unlabeled_dst, U_index, n_query, self.device)
                idx = selector.select(n_query)

            elif self.cfg.TRAINER.COOPAL.METHOD == "coreset":
                selector = Coreset(self.cfg, self.model, unlabeled_dst, U_index, val_x, n_query)
                idx = selector.select(n_query)

            elif self.cfg.TRAINER.COOPAL.METHOD == "pcb":
                selector = BADGE(self.cfg, self.model, unlabeled_dst, U_index, n_query, self.device)
                idx = selector.select(n_cand)
                if i != 0:
                    statistics = torch.zeros(self.num_classes)
                    for elem in dataset._train_x:
                        statistics[elem.label] += 1
                    selector = PCB(self.cfg, self.model, unlabeled_dst, idx, n_query, statistics, self.device)
                    idx = selector.select(n_query)
            else:
                print("NotImplementedError")
                idx= U_index

            if i != 0 and self.cfg.TRAINER.COOPAL.METHOD == "pcb":
                statistics = torch.zeros(self.num_classes)
                for elem in dataset._train_x:
                    statistics[elem.label] += 1
                selector = PCB(self.cfg, self.model, unlabeled_dst, idx, dataset.get_num_classes(unlabeled_dst), statistics, self.device)
                idx = selector.select(n_query)

            # =================================================
            if i != 0 and self.cfg.TRAINER.COOPAL.METHOD == "cbsq":
                for item in idx:
                    if isinstance(item, (tuple, list)) and len(item) == 3:
                        k, label, _ = item
                        sample_obj = copy.deepcopy(unlabeled_dst[k])
                        sample_obj._label = label
                        dataset._train_x.append(sample_obj)
                        strategy_train_x.append(sample_obj)
                        U_index.remove(k)
                    else:
                        k = int(item)
                        sample_obj = copy.deepcopy(unlabeled_dst[k])
                        dataset._train_x.append(sample_obj)
                        strategy_train_x.append(sample_obj)
                        U_index.remove(k)
            else:
                for k in idx:
                    sample_obj = copy.deepcopy(unlabeled_dst[k])
                    dataset._train_x.append(sample_obj)
                    strategy_train_x.append(sample_obj)
                    U_index.remove(k)

            print(f"Dataset._train_x : {len(dataset._train_x)}")

     
            # =================================================
            self.train_loader_x = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset._train_x,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=True),
                is_train=True,
                dataset_wrapper=None
            )


            if i == rounds:
                break

            self.acc = []
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            self.after_train()
            
            

            # =================================================
            # ROUND-BY-ROUND METRICS AND SUMMARIES
            # =================================================
            if len(self.acc) > 0:
                final_acc = self.acc[-1]
                round_data.append((final_acc, cluster_acc_train, cluster_acc_val, budget_saving, len_selected_indices, corr_ratio))
                print(f"\n>>> Round {i} Finished | Acc: {final_acc:.2f} | Time: {time.time() - start:.2f}s")


        # =========================================================
        # FINAL SUMMARY
        # =========================================================
        print("\n\n========================================================")
        print(f"=============== FINAL SUMMARY ({strategy.upper()}) ===============")
        print("========================================================")
        for r_idx, data in enumerate(round_data):
            print(f"Round {r_idx}: ACC Modelo: {data[0]:.2f} | "
                  f"TURTLE_train: {data[1]:.2f} | TURTLE_val: {data[2]:.2f} | "
                  f"Budget saving: {data[3]} | Selected: {data[4]} | "
                  f"Corr ratio: {data[5]:.4f}")


        output_dir = f"results/{self.cfg.DATASET.NAME_ADJ}/{strategy}/"
        os.makedirs(output_dir, exist_ok=True)

        csv_filename = os.path.join(output_dir, f"{strategy}_{self.cfg.SEED}.csv")

        headers = ["Round", "ACC", "TURTLE_train", "TURTLE_val", "Budget_saving", "Selected", "Corr_ratio"]
        
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r_idx, data in enumerate(round_data):
                writer.writerow(
                    [r_idx, f"{data[0]:.4f}", f"{data[1]:.4f}", f"{data[2]:.4f}",
                    data[3], f"{data[4]:.4f}", data[5]]
                )

        print(f"\nResults saved successfully to: {csv_filename}")