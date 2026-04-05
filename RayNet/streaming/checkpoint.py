"""
MinIO-backed checkpoint manager for RayNet training.

Saves and loads training checkpoints (model weights, optimizer state,
scheduler state, AMP scaler state, and full training metadata) to
S3-compatible object storage (MinIO).

Checkpoints are organized by run_id:

    s3://<bucket>/<prefix>/<run_id>/
        metadata.json          — run config + per-epoch metrics
        checkpoint_epoch5.pt   — periodic checkpoint
        checkpoint_epoch10.pt
        best_model.pt          — best validation checkpoint
        latest.pt              — most recent (for resume)

Usage:
    from RayNet.streaming.checkpoint import CheckpointManager

    mgr = CheckpointManager(
        bucket='raynet-checkpoints',
        endpoint='http://localhost:9000',
    )

    # During training
    mgr.save(epoch=5, model=model, optimizer=optimizer, ...)
    mgr.save_best(epoch=5, model=model, val_loss=0.42, ...)

    # Resume
    state = mgr.load_latest()          # or mgr.load('best_model.pt')
    model.load_state_dict(state['model_state_dict'])
"""

import io
import os
import json
import tempfile
import logging
from datetime import datetime, timezone

import torch

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    Minio = None
    S3Error = None

log = logging.getLogger(__name__)


def _get_model_state_dict(model):
    """Extract state_dict handling torch.compile wrapper."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


class CheckpointManager:
    """
    Manages training checkpoints on MinIO / S3-compatible storage.

    Each training run gets a unique run_id directory.  Checkpoints are
    saved locally first, then uploaded.  Loading downloads to a temp
    file and reads with torch.load.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix under the bucket (default 'checkpoints').
        run_id: Unique run identifier.  Generated from timestamp if None.
        endpoint: MinIO endpoint URL (default from S3_ENDPOINT_URL env var).
        access_key: S3 access key (default from AWS_ACCESS_KEY_ID env var).
        secret_key: S3 secret key (default from AWS_SECRET_ACCESS_KEY env var).
        local_cache: Local directory for temporary checkpoint files.
        save_local_copy: Keep a local copy after uploading to MinIO.
    """

    def __init__(
        self,
        bucket='raynet-checkpoints',
        prefix='checkpoints',
        run_id=None,
        endpoint=None,
        access_key=None,
        secret_key=None,
        local_cache='/tmp/raynet_ckpt',
        save_local_copy=True,
    ):
        assert Minio is not None, (
            "pip install minio  (required for MinIO checkpoint storage)"
        )

        self.bucket = bucket
        self.prefix = prefix
        self.run_id = run_id or datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.local_cache = local_cache
        self.save_local_copy = save_local_copy

        endpoint = endpoint or os.environ.get('S3_ENDPOINT_URL', 'http://localhost:9000')
        access_key = access_key or os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin')
        secret_key = secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin')

        secure = endpoint.startswith('https://')
        host = endpoint.replace('https://', '').replace('http://', '')

        self._client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        # Ensure bucket exists
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)
            log.info("Created bucket: %s", self.bucket)

        # Local run directory
        self._local_dir = os.path.join(local_cache, self.run_id)
        os.makedirs(self._local_dir, exist_ok=True)

        # In-memory metadata for this run
        self._metadata = {
            'run_id': self.run_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'epochs': {},
            'best': None,
            'config': {},
        }

        # Try loading existing metadata (for resume)
        self._try_load_metadata()

        log.info(
            "CheckpointManager: bucket=%s run=%s",
            self.bucket, self.run_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config(self, config):
        """
        Record training configuration metadata.

        Args:
            config: dict with training config (profile, backbone, lr, etc.)
        """
        self._metadata['config'] = config
        self._upload_metadata()

    def save(
        self,
        epoch,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        phase=None,
        train_metrics=None,
        val_metrics=None,
        extra=None,
        tag=None,
    ):
        """
        Save a training checkpoint and upload to MinIO.

        Args:
            epoch: Current epoch number.
            model: The model (handles torch.compile transparently).
            optimizer: Optimizer instance.
            scheduler: LR scheduler (optional).
            scaler: GradScaler for AMP (optional).
            phase: Training phase number (optional).
            train_metrics: Dict of training metrics for this epoch.
            val_metrics: Dict of validation metrics for this epoch.
            extra: Arbitrary extra data to include in the checkpoint.
            tag: Filename tag.  Defaults to 'checkpoint_epoch{epoch}'.
                 Use 'latest' for the resume checkpoint.

        Returns:
            S3 object key of the uploaded checkpoint.
        """
        filename = f"{tag or f'checkpoint_epoch{epoch}'}.pt"

        state = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': _get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'run_id': self.run_id,
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()
        if train_metrics:
            state['train_metrics'] = train_metrics
        if val_metrics:
            state['val_metrics'] = val_metrics
        if extra:
            state.update(extra)

        key = self._upload_checkpoint(filename, state)

        # Update metadata
        epoch_entry = {
            'epoch': epoch,
            'phase': phase,
            'filename': filename,
            'saved_at': state['saved_at'],
        }
        if train_metrics:
            epoch_entry['train_metrics'] = train_metrics
        if val_metrics:
            epoch_entry['val_metrics'] = val_metrics
        self._metadata['epochs'][str(epoch)] = epoch_entry
        self._upload_metadata()

        log.info("Saved checkpoint: %s", key)
        return key

    def save_best(
        self,
        epoch,
        model,
        val_loss,
        val_metrics=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        extra=None,
    ):
        """
        Save the best-model checkpoint (overwrites previous best).

        Args:
            epoch: Epoch that achieved the best val loss.
            model: The model.
            val_loss: Best validation loss value.
            val_metrics: Full validation metrics dict.
            optimizer: Optimizer (optional, included if provided).
            scheduler: LR scheduler (optional).
            scaler: GradScaler (optional).
            extra: Extra data.

        Returns:
            S3 object key.
        """
        state = {
            'epoch': epoch,
            'model_state_dict': _get_model_state_dict(model),
            'val_loss': val_loss,
            'run_id': self.run_id,
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        if val_metrics:
            state['val_metrics'] = val_metrics
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()
        if extra:
            state.update(extra)

        key = self._upload_checkpoint('best_model.pt', state)

        self._metadata['best'] = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'saved_at': state['saved_at'],
        }
        self._upload_metadata()

        log.info("Saved best model (epoch %d, val_loss=%.4f): %s", epoch, val_loss, key)
        return key

    def load(self, filename='latest.pt', map_location='cpu'):
        """
        Download and load a specific checkpoint.

        Args:
            filename: Checkpoint filename (e.g. 'best_model.pt', 'latest.pt').
            map_location: torch.load map_location argument.

        Returns:
            Checkpoint state dict.
        """
        key = self._object_key(filename)
        local_path = os.path.join(self._local_dir, filename)

        self._client.fget_object(self.bucket, key, local_path)
        state = torch.load(local_path, map_location=map_location, weights_only=False)
        log.info("Loaded checkpoint: %s (epoch %d)", key, state.get('epoch', -1))
        return state

    def load_latest(self, map_location='cpu'):
        """Load the most recent checkpoint for resuming training."""
        return self.load('latest.pt', map_location=map_location)

    def load_best(self, map_location='cpu'):
        """Load the best validation checkpoint."""
        return self.load('best_model.pt', map_location=map_location)

    def resume_state(self, model, optimizer, scheduler=None, scaler=None,
                     map_location='cpu'):
        """
        Convenience: load latest checkpoint and restore all state in-place.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to restore state.
            scheduler: LR scheduler to restore (optional).
            scaler: GradScaler to restore (optional).
            map_location: Device mapping.

        Returns:
            (start_epoch, checkpoint_dict) — start_epoch is the next
            epoch to train (checkpoint epoch + 1).
        """
        state = self.load_latest(map_location=map_location)

        # Handle torch.compile
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state['model_state_dict'])

        optimizer.load_state_dict(state['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])

        if scaler is not None and 'scaler_state_dict' in state:
            scaler.load_state_dict(state['scaler_state_dict'])

        start_epoch = state['epoch'] + 1
        log.info("Resumed from epoch %d (starting epoch %d)", state['epoch'], start_epoch)
        return start_epoch, state

    def list_checkpoints(self, run_id=None):
        """
        List all checkpoint files for a run.

        Args:
            run_id: Run to list (default: current run).

        Returns:
            List of object keys.
        """
        prefix = self._run_prefix(run_id or self.run_id) + '/'
        objects = self._client.list_objects(self.bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def list_runs(self):
        """
        List all run_ids in the bucket/prefix.

        Returns:
            List of run_id strings.
        """
        prefix = f"{self.prefix}/" if self.prefix else ''
        objects = self._client.list_objects(self.bucket, prefix=prefix)
        runs = set()
        for obj in objects:
            # Objects are prefix/run_id/filename — extract run_id
            parts = obj.object_name.replace(prefix, '').split('/')
            if parts and parts[0]:
                runs.add(parts[0])
        return sorted(runs)

    def get_metadata(self, run_id=None):
        """
        Download and return metadata.json for a run.

        Args:
            run_id: Run to query (default: current run).

        Returns:
            Metadata dict, or None if not found.
        """
        key = self._object_key('metadata.json', run_id=run_id)
        try:
            response = self._client.get_object(self.bucket, key)
            data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            return data
        except S3Error:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_prefix(self, run_id=None):
        run_id = run_id or self.run_id
        if self.prefix:
            return f"{self.prefix}/{run_id}"
        return run_id

    def _object_key(self, filename, run_id=None):
        return f"{self._run_prefix(run_id)}/{filename}"

    def _upload_checkpoint(self, filename, state):
        """Save state dict locally, then upload to MinIO."""
        local_path = os.path.join(self._local_dir, filename)
        torch.save(state, local_path)

        key = self._object_key(filename)
        self._client.fput_object(self.bucket, key, local_path)

        if not self.save_local_copy:
            os.remove(local_path)

        return key

    def _upload_metadata(self):
        """Serialize metadata to JSON and upload."""
        data = json.dumps(self._metadata, indent=2, default=str).encode('utf-8')
        key = self._object_key('metadata.json')
        self._client.put_object(
            self.bucket, key,
            io.BytesIO(data), len(data),
            content_type='application/json',
        )

        # Also save locally
        if self.save_local_copy:
            local_path = os.path.join(self._local_dir, 'metadata.json')
            with open(local_path, 'w') as f:
                json.dump(self._metadata, f, indent=2, default=str)

    def _try_load_metadata(self):
        """Load existing metadata if resuming a previous run."""
        existing = self.get_metadata()
        if existing is not None:
            self._metadata = existing
            log.info(
                "Loaded existing metadata for run %s (%d epochs recorded)",
                self.run_id, len(existing.get('epochs', {})),
            )
