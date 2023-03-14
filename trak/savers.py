from abc import ABC, abstractmethod
from typing import Optional, Iterable, Union
from pathlib import Path
import os
import uuid
import json
import numpy as np
from numpy.lib.format import open_memmap
import torch
ch = torch


class AbstractSaver(ABC):
    """
    Implementations of Saver class must implement getters and setters for TRAK
    features and scores, as well as intermediate values like gradients and
    "out-to-loss-gradient".

    The Saver class also handles the recording of metadata associated with each
    TRAK run. For example, hyperparameters like "JL dimension" -- the dimension
    used for the dimensionality reduction step of TRAk (Johnson-Lindenstrauss
    projection).
    """
    @abstractmethod
    def __init__(self,
                 save_dir: Union[Path, str],
                 metadata: Iterable) -> None:
        """ Creates the save directory if it doesn't already exist.
        If the save directory already exists, it validates that the current
        TRAKer class has the same hyperparameters (metadata) as the one
        specified in the save directory. Next, this method loads any existing
        computed results / intermediate values in the save directory.
        Last, it initalizes the self.current_* attributes which will be later
        populated with data for the "current" model ID of the TRAKer instance.

        Args:
            save_dir (Union[Path, str]): directory to save TRAK results,
                intermediate values, and metadata
            metadata (Iterable): a dictionary containing metadata related to the
                TRAKer class
        """
        self.metadata = metadata
        self.save_dir = Path(save_dir).resolve()
        os.makedirs(self.save_dir, exist_ok=True)

        # init TRAKer metadata
        self.metadata_file = self.save_dir.joinpath('metadata.json')
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                existsing_metadata = json.load(f)
            existing_jl_dim = int(existsing_metadata['JL dimension'])
            assert self.metadata['JL dimension'] == existing_jl_dim,\
                   f"In {self.save_dir} there are models using JL dimension {existing_jl_dim}\
                   , and this TRAKer instance uses JL dimension {self.metadata['JL dimension']}."

            existing_matrix_type = existsing_metadata['JL matrix type']
            assert self.metadata['JL matrix type'] == existing_matrix_type,\
                   f"In {self.save_dir} there are models using a {existing_matrix_type} JL matrix\
                   , and this TRAKer instance uses a {self.metadata['JL matrix type']} JL matrix."

        else:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)

        self.model_ids = {}
        # check if there are existing model ids in the save_dir
        self.model_ids_files = self.save_dir.rglob('id_*.json')

        for existing_model_id_file in self.model_ids_files:
            with open(existing_model_id_file, 'r') as f:
                existing_id = json.load(f)
                existing_id = {int(model_id): metadata
                               for model_id, metadata in existing_id.items()}
            self.model_ids.update(existing_id)
        # wlog set num_targets to those of a random model_id we could raise an
        # error here if different model_ids have different num_targets but this
        # could be a bit too stringent in some cases
        if len(self.model_ids.keys()) > 0:
            self.num_targets = list(self.model_ids.values())[0]['targets_scored']

        print(f'Existing IDs in {self.save_dir}: {list(self.model_ids.keys())}')

        self.current_model_id = None
        self.current_grads = None
        self.current_out_to_loss = None
        self.current_features = None
        self.current_target_grads = None

    @abstractmethod
    def register_model_id(self, model_id: int) -> None:
        """ Create metadata for a new model ID (checkpoint).

        Args:
            model_id (int): a unique ID for a checkpoint
        """
        ...

    @abstractmethod
    def serialize_model_id_metadata(self, model_id: int) -> None:
        """ Write to disk / commit any updates to the metadata associated
        to a given model ID

        Args:
            model_id (int): a unique ID for a checkpoint
        """
        ...

    @abstractmethod
    def load_store(self, model_id: int) -> None:
        """ Populates the self.current_* attributes with data for the
        given model ID (checkpoint).

        Args:
            model_id (int): a unique ID for a checkpoint
        """
        ...

    @abstractmethod
    def load_target_store(self,
                          model_id: int,
                          num_targets: int,
                          mode: Optional[str]) -> None:
        """ Loads/initializes metadata and store arrays for target results and
        features (gradients)

        Args:
            model_id (int): a unique ID for a checkpoint
            num_targets (int): number of target samples that will be scored
        """
        ...

    @abstractmethod
    def del_grads(self, model_id: int, target: bool) -> None:
        """ Delete the intermediate values (gradients) for a given model id

        Args:
            model_id (int): a unique ID for a checkpoint
            target (bool): if True, delete the gradients of the target samples,
                otherwise delete the train set gradients.
        """
        ...

    @abstractmethod
    def clear_target_grad_count(self, model_id: int) -> None:
        """ Reset the count for how many targets we are scoring

        Args:
            model_id (int): a unique ID for a checkpoint
        """
        ...


class ModelIDException(Exception):
    """ A minimal custom exception for errors related to model IDs """
    pass


class MmapSaver(AbstractSaver):
    """ A saver that uses memory-mapped numpy arrays. This makes small reads and
    writes (e.g.) during featurizing feasible without loading the entire file
    into memory.

    """
    def __init__(self, save_dir, metadata, train_set_size, proj_dim) -> None:
        super().__init__(save_dir=save_dir, metadata=metadata)
        self.train_set_size = train_set_size
        self.proj_dim = proj_dim

    def register_model_id(self,
                          model_id: int,
                          _allow_featurizing_already_registered: bool) -> None:
        """ This method
        1) checks if the model ID already exists in the save dir
        2) if yes, it raises an error since model IDs must be unique
        3) if not, it creates a metadata file for it and initalizes store mmaps

        Args:
            model_id (int): a unique ID for a checkpoint

        Raises:
            ModelIDException: raised if the model ID to be registered already
            exists
        """
        self.current_model_id = model_id

        if self.current_model_id in self.model_ids.keys() and (not _allow_featurizing_already_registered):
            err_msg = f'model id {self.current_model_id} is already registered. Check {self.save_dir}'
            raise ModelIDException(err_msg)
        self.model_ids[self.current_model_id] = {'featurized': 0,
                                                 'finalized': 0,
                                                 'targets_scored': 0}

        self.init_store(self.current_model_id)
        self.serialize_model_id_metadata(self.current_model_id)

    def serialize_model_id_metadata(self, model_id) -> None:
        with open(self.save_dir.joinpath(f'id_{model_id}.json'), 'w+') as f:
            content = {self.current_model_id: self.model_ids[self.current_model_id]}
            json.dump(content, f)

    def init_store(self, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        try:
            os.makedirs(prefix)
        except FileExistsError:
            raise ModelIDException(f'model id folder {prefix} already exists')

        self.load_store(model_id, mode='w+')

    def load_store(self,
                   model_id: int,
                   mode: Optional[str] = 'r+') -> None:
        """ This method uses numpy memmaps for serializing the TRAK results and
        intermediate values.

        Args:
            model_id (int): a unique ID for a checkpoint
            mode (str, optional): Defaults to 'r+'.
        """
        self.current_model_id = model_id
        prefix = self.save_dir.joinpath(str(model_id))

        self.current_grads_path = prefix.joinpath('grads.mmap')
        self.current_grads = open_memmap(filename=self.current_grads_path,
                                         mode=mode,
                                         shape=(self.train_set_size, self.proj_dim),
                                         dtype=np.float32)

        self.current_out_to_loss_path = prefix.joinpath('out_to_loss.mmap')
        self.current_out_to_loss = open_memmap(filename=self.current_out_to_loss_path,
                                               mode=mode,
                                               shape=(self.train_set_size, 1),
                                               dtype=np.float32)

        self.current_features_path = prefix.joinpath('features.mmap')
        self.current_features = open_memmap(filename=self.current_features_path,
                                            mode=mode,
                                            shape=(self.train_set_size, self.proj_dim),
                                            dtype=np.float32)

    def load_target_store(self,
                          model_id: int,
                          num_targets: int,
                          mode: Optional[str] = 'r+',
                          ) -> None:
        """_summary_

        Args:
            model_id (int): _description_
            num_targets (int): _description_
        """
        self.num_targets = num_targets
        self.current_model_id = model_id
        self.model_ids[model_id]['targets_scored'] = self.num_targets
        self.serialize_model_id_metadata(model_id)

        prefix = self.save_dir.joinpath(str(model_id))

        self.current_target_grads_path = prefix.joinpath('grads_target.mmap')
        self.current_target_grads = open_memmap(filename=self.current_target_grads_path,
                                                mode=mode,
                                                shape=(num_targets, self.proj_dim),
                                                dtype=np.float32)

    def save_scores(self, scores, exp_name):
        prefix = self.save_dir.joinpath('scores')
        if exp_name is None:
            exp_name = str(uuid.uuid4())  # generate random unique ID
            print(f'Saving scores in {prefix}/scores_{exp_name}.npy')
        filename = 'scores_' + exp_name + '.npy'
        if not os.path.isdir(prefix):
            os.makedirs(prefix)
        np.save(prefix.joinpath(filename), scores)

    def del_grads(self, model_id, target=False):
        if target:
            grads_file = self.save_dir.joinpath(str(model_id)).joinpath('grads_target.mmap')
            self.clear_target_grad_count(model_id)
        else:
            grads_file = self.save_dir.joinpath(str(model_id)).joinpath('grads.mmap')

        # delete grads memmap
        grads_file.unlink()

    def clear_target_grad_count(self, model_id):
        self.model_ids[model_id]['targets_scored'] = 0
        self.serialize_model_id_metadata(model_id)

        if model_id == self.current_model_id:
            self.current_num_target_grads = 0
