from tools import *
from ModelZoo.example.model_train import Net_search
from ModelZoo.example.model_test import Net
class LightningNet(pl.LightningModule):
    def __init__(self, trial=None):
        super(LightningNet, self).__init__()
        # build model
        if trial is None:
            self.model = Net()
        else:
            self.model = Net_search(trial=trial)

    def forward(self, data):
        return self.model(data)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def prepare_data(self):
        # transforms for images
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd()+'Dataset', train=True, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64)


    @staticmethod
    def add_model_specific_args(model_name, dataset_name):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = HyperOptArgumentParser()

        # arguments from dataset
        if dataset_name == 'fish_body':
            parser.add_argument('--data_name', type=str, default='fish_body', help='data_set name')

        demo_log_dir = os.path.join(root_dir, 'Logs/' + model_name + '_' + dataset_name + '_logs')

        checkpoint_dir = os.path.join(demo_log_dir, model_name + '_' + dataset_name + '_model')
        loss_log_dir = os.path.join(demo_log_dir, model_name + '_' + dataset_name + 'loss_data')

        parser.add_argument('--loss_save_path', type=str, default=loss_log_dir, help='where to save logs')
        parser.add_argument('--model_save_path', type=str, default=checkpoint_dir, help='where to save model')

        parser.add_argument('--data_dir', default='./Dataset', type=str)
        parser.add_argument("--cuda", default=1, type=int,
                            help="set it to 1 for running on GPU, 0 for CPU")

        return parser