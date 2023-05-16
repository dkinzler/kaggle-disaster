import os
from typing import Optional
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class DisasterApp:
    """Wraps a model and optimizer and provides functionality to
    run the training loop, create predictions for a test dataset, perform inference,
    save and load checkpoints and log training metrics to Tensorboard.
    """
    def __init__(self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            device: Optional[torch.device] = None,
            checkpoint_name: str = "model",
            checkpoint_dir: Optional[str] = None,
            tensorboard_log_dir: Optional[str] = None) -> None:

        self.model = model
        self.optimizer = optimizer
        if optimizer is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if device is None:
            self.device = self._get_default_device()
        else:
            self.device = device

        self.model.to(self.device)

        self.epoch = 0
        self.global_step = 0

        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = checkpoint_dir

        self._tensorboard_summary_writer = None
        if tensorboard_log_dir is not None:
            self._tensorboard_summary_writer = SummaryWriter(tensorboard_log_dir, flush_secs=30)

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def train(self,
             train_data,
             val_data=None,
          num_epochs: int = 20,
          profiler: Optional[torch.profiler.profile] = None,
          log_freq: int = 10,
          checkpoint_freq: Optional[int] = None,
       ) -> None:
        "Train the model using the given data."
        if self.optimizer is None:
            raise TypeError("optimizer is None")

        for epoch in range(num_epochs):
            # need to set training mode in every epoch, since _validate might change to eval mode
            self.model.train()
            for i, (inputs, targets) in enumerate(train_data):
                loss, accuracy = self._train_step(inputs, targets)

                if profiler is not None:
                    profiler.step()

                if i % log_freq == 0:
                    print(f"epoch: {self.epoch}  step {i}  loss: {loss}  accuracy: {accuracy}")
                    self._log_metrics(loss, accuracy, tag="train")

            # eval at end of an epoch
            if val_data is not None:
                loss, accuracy = self._validate(val_data)
                print(f"validating at epoch {self.epoch}  loss: {loss}  accuracy: {accuracy}")
                self._log_metrics(loss, accuracy, tag="validation")

            if (checkpoint_freq is not None) and (checkpoint_freq > 0):
                if (epoch+1) % checkpoint_freq == 0:
                    self.save_checkpoint()

            self.epoch += 1

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        return loss.item(), self._accuracy(predictions, targets)

    def _log_metrics(self, loss: float, accuracy: float, tag: str = "train") -> None:
        if self._tensorboard_summary_writer is not None:
            self._tensorboard_summary_writer.add_scalars("loss", {tag: loss}, global_step=self.global_step)
            self._tensorboard_summary_writer.add_scalars("accuracy", {tag: accuracy}, global_step=self.global_step)

    def _validate(self, data) -> tuple[float, float]:
        self.model.eval()

        num_samples = 0
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in data:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(inputs)
                loss = F.cross_entropy(predictions, targets, reduction="sum")

                num_samples += targets.size(dim=0)
                total_loss += loss.item()
                correct += self._count_correct_predictions(predictions, targets)
    
        return total_loss/num_samples, correct/num_samples

    def test(self, data) -> dict[str, int]:
        """Predicts labels for the given data samples, returns a map from
        sample id to label.
        """
        self.model.eval()

        id_to_pred = {}
        with torch.no_grad():
            for ids, inputs in data:
                inputs = inputs.to(self.device)

                pred = self.model(inputs)
                classes = torch.argmax(pred, dim=1)
                for j in range(inputs.size(dim=0)):
                    id_to_pred[ids[j]] = classes[j].item()
        
        return id_to_pred

    def inference(self, data: torch.Tensor) -> int:
        """Return label prediction for the given data sample."""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            prediction = torch.argmax(self.model(data), dim=0)
            return prediction.item()

    def _accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            return self._count_correct_predictions(predictions, targets) / float(targets.size(dim=0))

    def _count_correct_predictions(self, predictions: torch.Tensor, targets: torch.Tensor) -> int:
        with torch.no_grad():
            return torch.sum(torch.argmax(predictions, dim=1) == targets).item()

    def save_checkpoint(self) -> None:
        """Save a checkpoint of the current app state, which consists of
        the model, optimizer, current epoch and global step."""
        checkpoint_dir = self.checkpoint_dir if self.checkpoint_dir is not None else ""
        name = f"{self.checkpoint_name}-{self.epoch}.pt"
        filepath = os.path.join(checkpoint_dir, name)

        if checkpoint_dir != "":
            os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save({
            "model": self.model,
            "optimizer": self.optimizer,
            "epoch": self.epoch+1,
            "global_step": self.global_step,
        }, filepath)

        print(f"saved checkpoint: {filepath}")

def load_checkpoint(checkpoint: str, inference: bool = False, device: Optional[torch.device] = None, checkpoint_name: Optional[str] = None, checkpoint_dir: Optional[str] = None, tensorboard_log_dir: Optional[str] = None) -> DisasterApp:
    """Load the given checkpoint file.
    If inference is set to True, only the model will be loaded not the optimizer.
    """
    ckpt = torch.load(checkpoint)
    model = ckpt["model"]
    optimizer = None
    if not inference:
        optimizer = ckpt["optimizer"]
    app = DisasterApp(model=model, optimizer=optimizer, device=device, checkpoint_name=checkpoint_name, checkpoint_dir=checkpoint_dir, tensorboard_log_dir=tensorboard_log_dir)
    app.epoch = ckpt["epoch"]
    app.global_step = ckpt["global_step"]
    return app
