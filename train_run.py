from ModelZoo.model_structure import *

monitor_value = 'avg_val_loss'

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def objective(trial):
    # 2. log every result in Experience for grid search
    loss_logger = TestTubeLogger(
        model_para.loss_save_path,
        name='model_{}'.format(model_name)
    )

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=model_para.model_save_path,
        save_top_k=True,
        verbose=True,
        monitor=monitor_value,
        prefix='best'
    )


    metrics_callback = MetricsCallback()

    # 4. train
    trainer = pl.Trainer(
        gpus=1,
        logger=loss_logger,
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor=monitor_value),
        checkpoint_callback=checkpoint_callback,
        callbacks=[metrics_callback],
        max_epochs=5
    )
    model = LightningNet(trial)
    trainer.fit(model)

    return metrics_callback.metrics[-1][monitor_value]


def run():
    ########### 定义剪枝规则 ###########
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)

    ########### start training ###########
    study.optimize(objective, n_trials=3)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    ########### visualization ###########
    print("ssh -L 16006:127.0.0.1:6006 username@remoteIP")
    print("conda activate SizeMeasure")
    print('View tensorboard logs by running\n tensorboard --logdir %s' % model_para.loss_save_path)
    print('and going to http://localhost:16006 on your browser')

if __name__ == '__main__':
    model_name = 'example'
    dataset_name = 'minist'

    # 1. get model parameter
    parser = LightningNet.add_model_specific_args(model_name, dataset_name)
    model_para = parser.parse_args()

    run()