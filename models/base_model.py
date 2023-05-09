import pytorch_lightning as pl
import torch.optim
from toolbox.utils import get_lr


class DummyClass(pl.LightningModule):
    def __init__(self, batch_size=None, sync_dist=True):
        super().__init__()

        self.use_metric = False
        self._metric_function = None
        self.sync_dist = sync_dist

        self.batch_size = batch_size

        self.std_dict = dict()

    def attach_metric_function(self, metric_function, start_using_metric=True):
        if start_using_metric:
            self.use_metric = True
        self._metric_function = metric_function

    def log(
        self, name, value, batch_size=None, **kwargs
    ):  # Overload for batch_size passing in masked tensors
        if batch_size is None:
            batch_size = self.batch_size
        super().log(name, value, batch_size=batch_size, rank_zero_only=True, **kwargs)

    def log_metric_old(self, prefix, sync_dist=None, **kwargs):
        if self.use_metric and self._metric_function is not None:
            value_dict = self._metric_function(**kwargs)
            if sync_dist is None:
                sync_dist = self.sync_dist
            for key, value in value_dict.items():
                self.log(f"{prefix}/metrics/{key}", value, sync_dist=sync_dist)

    def log_metric(self, prefix, sync_dist=None, **kwargs):
        if self.use_metric and self._metric_function is not None:
            value_dict = self._metric_function(**kwargs)
            for key, value in value_dict.items():
                if (
                    key[:-4] == "_std"
                ):  # If it's an std metric, delete it. Then go to next entry
                    del value_dict[key]
                    continue
                if key not in self.std_dict:
                    self.std_dict[key] = []
                self.std_dict[key].append(value)
            if sync_dist is None:
                sync_dist = self.sync_dist
            for key, value in value_dict.items():
                self.log(f"{prefix}/metrics/{key}", value, sync_dist=sync_dist)
            for key, value in self.std_dict.items():
                self.log(
                    f"{prefix}/metrics/{key}",
                    torch.std(torch.tensor(value)),
                    sync_dist=sync_dist,
                )
                print(f"{prefix}/metrics/{key}", torch.std(torch.tensor(value)))


class GNN_Abstract_Base_Class(DummyClass):
    def __init__(self, model, optim_args, batch_size=None, sync_dist=True):
        super().__init__(batch_size=batch_size, sync_dist=sync_dist)

        self.model = model
        self.initial_lr = optim_args["lr"]
        self.scheduler_args = {
            "patience": optim_args["scheduler_step"],
            "factor": optim_args["scheduler_factor"],
        }
        self.scheduler_monitor = optim_args["monitor"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, **(self.scheduler_args)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.scheduler_monitor,
        }

    def _log_lr(self) -> None:
        optim = self.optimizers()
        if optim:
            lr = get_lr(optim)
            self.log("lr", lr)

    def on_train_epoch_start(self) -> None:
        self._log_lr()
