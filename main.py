from pathlib import Path
import numpy as np

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader
import wandb

from options import Options
from data_utils import MILDataset, MILDatasetIndices, get_multi_cohort_df
from classifier import ClassifierLightning


def main(args):
    args.random_state = torch.randint(0, 1000, (1, )).item() if args.random_state is None else args.random_state
    pl.seed_everything(args.random_state, workers=True)

    # saving locations
    base_path = Path(args.save_dir)  # adapt to own target path
    logging_name = f'{args.name}_{args.model}_{"-".join(args.cohorts)}_{args.norm}_{"-".join(args.target_labels)}' if not args.debug else 'debug'
    base_path = base_path / logging_name
    model_path = base_path / 'models'
    output_path = base_path / 'outputs'
    fold_path = base_path / 'folds'
    results_path = base_path / 'results'
    logging_name = f'{args.name}_{args.model}_{"-".join(args.cohorts)}_{args.norm}_{"-".join(args.target_labels)}' if not args.debug else 'debug'

    norm_val = 'raw' if args.norm in ['histaugan', 'efficient_histaugan'] else args.norm
    norm_test = 'raw' if args.norm in ['histaugan', 'efficient_histaugan'] else args.norm

    # --------------------------------------------------------
    # load data
    # --------------------------------------------------------
    print('\n--- load dataset ---')
    categories = ['Not mut.', 'Mutat.', 'nonMSIH', 'MSIH', 'WT', 'MUT', 'wt', 'MT']
    data = get_multi_cohort_df(
        args.cohorts, [args.target], categories, norm=args.norm, feats=args.feats
    )

    dataset_ext = MILDataset(
        [args.ext_cohort], [args.target],
        categories,
        norm=norm_test,
        feats=args.feats,
        clini_info=args.clini_info
    )
    test_ext_dataloader = DataLoader(
        dataset=dataset_ext, batch_size=1, shuffle=False, pin_memory=True
    )

    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    patient_df = data.groupby('PATIENT').first().reset_index()
    target_stratisfy = args.target if type(args.target) is str else args.target[0]
    splits = skf.split(patient_df, patient_df[target_stratisfy])

    for k, (train_val_idxs, test_idxs) in enumerate(splits):
        train_idxs, val_idxs = train_test_split( train_val_idxs, stratify=patient_df.iloc[train_val_idxs][target_stratisfy], random_state=args.random_state)
        
        # training dataset
        train_dataset = MILDatasetIndices(
            data,
            train_idxs, [args.target],
            num_tiles=args.num_tiles,
            pad_tiles=args.pad_tiles,
            norm=args.norm
        )
        np.savetxt(fold_path / f'folds_{logging_name}_fold{k}_train.csv', train_idxs, delimiter=',')
        print(f'num training samples in fold {k}: {len(train_dataset)}')
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=args.bs, shuffle=True, pin_memory=True
        )
        
        # validation dataset
        val_dataset = MILDatasetIndices(data, val_idxs, [args.target], norm=norm_val)
        np.savetxt(fold_path / f'folds_{logging_name}_fold{k}_val.csv', val_idxs, delimiter=',')
        print(f'num validation samples in fold {k}: {len(val_dataset)}')
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=args.bs, shuffle=True, pin_memory=True
        )

        # test dataset (in-domain)
        test_dataset = MILDatasetIndices(data, test_idxs, [args.target], norm=norm_test)
        np.savetxt(fold_path / f'folds_{logging_name}_fold{k}_test.csv', test_idxs, delimiter=',')
        print(f'num test samples in fold {k}: {len(test_dataset)}')
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=args.bs, shuffle=False, pin_memory=True
        )

        # idx=2 since the ouput is feats, coords, labels
        num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])
        args.pos_weight = torch.Tensor((len(train_dataset) - num_pos) / num_pos).item()
        args.criterion = "BCEWithLogitsLoss"

        # --------------------------------------------------------
        # model
        # --------------------------------------------------------
        model = ClassifierLightning(args)

        # --------------------------------------------------------
        # logging
        # --------------------------------------------------------
        logger = WandbLogger(
            project=args.project,
            name=f'{logging_name}_fold{k}',
            save_dir=args.log_dir,
            reinit=True,
            settings=wandb.Settings(start_method='fork'),
        )

        csv_logger = CSVLogger(
            save_dir=args.log_dir,
            name=f'{logging_name}_fold{k}',
        )

        # --------------------------------------------------------
        # model saving
        # --------------------------------------------------------
        checkpoint_callback = ModelCheckpoint(
            monitor='auroc/val' if args.stop_criterion == 'auroc' else 'loss/val',
            dirpath=model_path,
            filename=f'best_model_{logging_name}_fold{k}',
            save_top_k=1,
            mode='max' if args.stop_criterion == 'auroc' else 'min',
        )

        trainer = pl.Trainer(
            logger=[logger, csv_logger],
            accelerator='auto',
            precision=16,
            accumulate_grad_batches=4,
            gradient_clip_val=1,
            callbacks=[checkpoint_callback],
            # track_grad_norm=2,      # debug
            # num_sanity_val_steps=0,  # debug
            # val_check_interval=0.1,  # debug
            # limit_val_batches=0.1,  # debug
            # limit_train_batches=6,  # debug
            # limit_val_batches=6,    # debug
            # log_every_n_steps=1,  # debug
            # fast_dev_run=True,    # debug
            # max_epochs=1,         # debug
            # max_steps=6,          # debug
            # enable_model_summary=False,  # debug
        )

        results_val = trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_path=args.resume,
        )
        logger.log_table('results/val', results_val)

        results_test = trainer.test(
            model.load_from_checkpoint(checkpoint_callback.best_model_path),
            [test_dataloader, test_ext_dataloader],
        )
        test_cohorts = [args.cohorts, *args.ext_cohort]
        for i, result in enumerate(results_test):
            logger.log_table(f'results/test_{i}', result)


if __name__ == '__main__':
    parser = Options()
    args = parser.parse()
    
    main(args)
