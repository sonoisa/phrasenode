from datetime import datetime

from torch.nn.utils import clip_grad_norm_

from gtd.ml.training_run import TrainingRun
from gtd.ml.torch.checkpoints import Checkpoints
from gtd.utils import cached_property
from gtd.ml.torch.utils import isfinite


class TorchTrainingRun(TrainingRun):
    def __init__(self, config, save_dir):
        super(TorchTrainingRun, self).__init__(config, save_dir)
        self.workspace.add_dir('checkpoints', 'checkpoints')

    @cached_property
    def checkpoints(self):
        return Checkpoints(self.workspace.checkpoints)

    @classmethod
    def _finite_grads(cls, parameters):
        """Check that all parameter gradients are finite.

        Args:
            parameters (List[Parameter])

        Return:
            bool
        """
        for param in parameters:
            if param.grad is None:
                continue
            if not isfinite(param.grad.detach().sum()):
                return False
        return True

    @classmethod
    def _take_grad_step(cls, train_state, loss, max_grad_norm=float('inf')):
        """Try to take a gradient step w.r.t. loss.
        
        If the gradient is finite, takes a step. Otherwise, does nothing.
        
        Args:
            train_state (TrainState)
            loss (Variable): a differentiable scalar variable
            max_grad_norm (float): gradient norm is clipped to this value.
        
        Returns:
            finite_grads (bool): True if the gradient was finite.
            grad_norm (float): norm of the gradient (BEFORE clipping)
        """
        model, optimizer = train_state.model, train_state.optimizer
        optimizer.zero_grad()
        loss.backward()

        # clip according to the max allowed grad norm
        grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
        # (this returns the gradient norm BEFORE clipping)

        # track the gradient norm over time
        train_state.track_grad_norms(grad_norm)

        finite_grads = cls._finite_grads(model.parameters())

        # take a step if the grads are finite
        if finite_grads:
            optimizer.step()

        # increment step count
        train_state.increment_train_steps()

        return finite_grads, grad_norm

    def _log_stats(self, stats, step):
        """Log stats to Tensorboard and metadata.
        
        Args:
            stats (dict[tuple[str], float]): a map from a stat name (expressed as a string tuple) to a float
            step (int): training step that we are on, for Tensorboard plots
        """
        for path, val in stats.items():
            # log to TBoard
            name = '_'.join(path)
            self.tb_logger.log_value(name, val, step)

            # log to metadata
            with self.metadata.name_scope_path(['stats'] + list(path[:-1])):
                self.metadata[path[-1]] = val

    def _update_metadata(self, train_state):
        self.metadata['last_seen'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['steps'] = train_state.train_steps
        self.metadata['max_grad_norm'] = train_state.max_grad_norm
