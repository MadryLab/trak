"""
This module contains classes that save TRAK results, intermediate values, and
metadata to disk. The :code:`AbstractSaver` class defines the interface for
savers. Then, we provide one implementation:
- :class:`MmapSaver`: A saver that uses memory-mapped numpy arrays. This makes
  loading and saving small chunks of data (e.g.) during featurizing feasible
  without loading the entire file into memory.
"""
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Union
from pathlib import Path
import os
import logging
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
    used for the dimensionality reduction step of TRAK (Johnson-Lindenstrauss
    projection).
    """

    @abstractmethod
    def __init__(
        self,
        save_dir: Union[Path, str],
        metadata: Iterable,
        load_from_save_dir: bool,
        logging_level: int,
        use_half_precision: bool,
    ) -> None:
        """Creates the save directory if it doesn't already exist.
        If the save directory already exists, it validates that the current
        TRAKer class has the same hyperparameters (metadata) as the one
        specified in the save directory. Next, this method loads any existing
        computed results / intermediate values in the save directory. Last, it
        initalizes the self.current_store attributes which will be later
        populated with data for the "current" model ID of the TRAKer instance.

        Args:
            save_dir (Union[Path, str]): directory to save TRAK results,
                intermediate values, and metadata
            metadata (Iterable): a dictionary containing metadata related to the
                TRAKer class
            load_from_save_dir (bool): If True, the Saver instance will attempt
                to load existing metadata from save_dir. May lead to I/O issues
                if multiple Saver instances ran in parallel have this flag set
                to True. See the SLURM tutorial in our docs for more details.
            logging_level (int):
                logging level for the logger associated with this Saver instance
            use_half_precision (bool):
                If True, the Saver instance will save all results and intermediate
                values in half precision (float16).
        """
        self.metadata = metadata
        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir
        self.use_half_precision = use_half_precision

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir.joinpath("scores"), exist_ok=True)

        self.logger = logging.getLogger("STORE")
        self.logger.setLevel(logging_level)

        # init TRAKer metadata
        self.metadata_file = self.save_dir.joinpath("metadata.json")
        if os.path.exists(self.metadata_file) and self.load_from_save_dir:
            with open(self.metadata_file, "r") as f:
                existsing_metadata = json.load(f)
            existing_jl_dim = int(existsing_metadata["JL dimension"])
            assert (
                self.metadata["JL dimension"] == existing_jl_dim
            ), f"In {self.save_dir} there are models using JL dimension {existing_jl_dim},\n\
                   and this TRAKer instance uses JL dimension {self.metadata['JL dimension']}."

            existing_matrix_type = existsing_metadata["JL matrix type"]
            assert (
                self.metadata["JL matrix type"] == existing_matrix_type
            ), f"In {self.save_dir} there are models using a {existing_matrix_type} JL matrix,\n\
                   and this TRAKer instance uses a {self.metadata['JL matrix type']} JL matrix."

            assert (
                self.metadata["train set size"] == existsing_metadata["train set size"]
            ), f"In {self.save_dir} there are models TRAKing\n\
                   {existsing_metadata['train set size']} examples, and in this TRAKer instance\n\
                   there are {self.metadata['train set size']} examples."

        elif self.load_from_save_dir:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f)

        self.model_ids = {}
        self.experiments = {}
        self.experiments_file = self.save_dir.joinpath("experiments.json")
        if self.load_from_save_dir:
            # check if there are existing model ids in the save_dir
            self.model_ids_files = self.save_dir.rglob("id_*.json")

            for existing_model_id_file in self.model_ids_files:
                with open(existing_model_id_file, "r") as f:
                    existing_id = json.load(f)
                    existing_id = {
                        int(model_id): metadata
                        for model_id, metadata in existing_id.items()
                    }
                self.model_ids.update(existing_id)

            if os.path.isfile(self.experiments_file):
                with open(self.experiments_file, "r") as f:
                    self.experiments.update(json.load(f))
            else:
                with open(self.experiments_file, "w") as f:
                    json.dump({}, f)

        existing_ids = list(self.model_ids.keys())
        if len(existing_ids) > 0:
            self.logger.info(
                f"Existing model IDs in {self.save_dir}: {sorted(existing_ids)}"
            )
            ids_finalized = sorted(
                list([id for id, v in self.model_ids.items() if v["is_finalized"] == 1])
            )
            if len(ids_finalized) > 0:
                self.logger.info(f"Model IDs that have been finalized: {ids_finalized}")
            else:
                self.logger.info(
                    f"No model IDs in {self.save_dir} have been finalized."
                )
        else:
            self.logger.info(f"No existing model IDs in {self.save_dir}.")

        if len(list(self.experiments.keys())) > 0:
            self.logger.info("Existing TRAK scores:")
            for exp_name, values in self.experiments.items():
                self.logger.info(f"{exp_name}: {values['scores_path']}")
        else:
            self.logger.info(f"No existing TRAK scores in {self.save_dir}.")

        self.current_model_id = None
        self.current_store = {
            "grads": None,
            "out_to_loss": None,
            "features": None,
        }

    @abstractmethod
    def register_model_id(self, model_id: int) -> None:
        """Create metadata for a new model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint

        """
        ...

    @abstractmethod
    def serialize_current_model_id_metadata(self) -> None:
        """Write to disk / commit any updates to the metadata associated
        to the current model ID

        """
        ...

    @abstractmethod
    def init_store(self, model_id: int) -> None:
        """Initializes store for a given model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint
        """
        ...

    @abstractmethod
    def init_experiment(self, model_id: int) -> None:
        """Initializes store for a given experiment & model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint
        """
        ...

    @abstractmethod
    def load_current_store(self, model_id: int) -> None:
        """Populates the self.current_store attributes with data for the
        given model ID (checkpoint).

        Args:
            model_id (int):
                a unique ID for a checkpoint

        """
        ...

    @abstractmethod
    def save_scores(self, exp_name: str) -> None:
        """Saves scores for a given experiment name

        Args:
            exp_name (str):
                experiment name

        """
        ...

    @abstractmethod
    def del_grads(self, model_id: int, target: bool) -> None:
        """Delete the intermediate values (gradients) for a given model id

        Args:
            model_id (int):
                a unique ID for a checkpoint
            target (bool):
                if True, delete the gradients of the target samples, otherwise
                delete the train set gradients.

        """
        ...


class ModelIDException(Exception):
    """A minimal custom exception for errors related to model IDs"""

    pass


class MmapSaver(AbstractSaver):
    """A saver that uses memory-mapped numpy arrays. This makes small reads and
    writes (e.g.) during featurizing feasible without loading the entire file
    into memory.

    """

    def __init__(
        self,
        save_dir,
        metadata,
        train_set_size,
        proj_dim,
        load_from_save_dir,
        logging_level,
        use_half_precision,
    ) -> None:
        super().__init__(
            save_dir=save_dir,
            metadata=metadata,
            load_from_save_dir=load_from_save_dir,
            logging_level=logging_level,
            use_half_precision=use_half_precision,
        )
        self.train_set_size = train_set_size
        self.proj_dim = proj_dim

    def register_model_id(
        self, model_id: int, _allow_featurizing_already_registered: bool
    ) -> None:
        """This method
        1) checks if the model ID already exists in the save dir
        2) if yes, it raises an error since model IDs must be unique
        3) if not, it creates a metadata file for it and initalizes store mmaps

        Args:
            model_id (int):
                a unique ID for a checkpoint

        Raises:
            ModelIDException:
                raised if the model ID to be registered already exists

        """
        self.current_model_id = model_id

        if self.current_model_id in self.model_ids.keys() and (
            not _allow_featurizing_already_registered
        ):
            err_msg = f"model id {self.current_model_id} is already registered. Check {self.save_dir}"
            raise ModelIDException(err_msg)
        self.model_ids[self.current_model_id] = {"is_featurized": 0, "is_finalized": 0}

        self.init_store(self.current_model_id)
        self.serialize_current_model_id_metadata(already_exists=False)

    def serialize_current_model_id_metadata(self, already_exists=True) -> None:
        is_featurized = int(
            self.current_store["is_featurized"].sum() == self.train_set_size
        )

        # update the metadata JSON file
        content = {
            self.current_model_id: {
                "is_featurized": is_featurized,
                "is_finalized": self.model_ids[self.current_model_id]["is_finalized"],
            }
        }
        # update the metadata dict within the class instance
        self.model_ids[self.current_model_id]["is_featurized"] = is_featurized
        if (is_featurized == 1) or not already_exists:
            with open(
                self.save_dir.joinpath(f"id_{self.current_model_id}.json"), "w"
            ) as f:
                json.dump(content, f)

    def init_store(self, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        if os.path.exists(prefix):
            self.logger.info(f"Model ID folder {prefix} already exists")
        os.makedirs(prefix, exist_ok=True)
        featurized_so_far = np.zeros(shape=(self.train_set_size,), dtype=np.int32)
        ft = self._load(
            prefix.joinpath("_is_featurized.mmap"),
            shape=(self.train_set_size,),
            mode="w+",
            dtype=np.int32,
        )
        ft[:] = featurized_so_far[:]
        ft.flush()

        self.load_current_store(model_id, mode="w+")

    def init_experiment(self, exp_name, num_targets, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        if not os.path.exists(prefix):
            raise ModelIDException(
                f"model ID folder {prefix} does not exist,\n\
            cannot start scoring"
            )
        self.experiments[exp_name] = {
            "num_targets": num_targets,
            "scores_path": str(self.save_dir.joinpath(f"scores/{exp_name}.mmap")),
            "scores_finalized": 0,
        }

        # update experiments.json
        with open(self.experiments_file, "r") as fp:
            exp_f = json.load(fp)

        exp_f[exp_name] = self.experiments[exp_name]
        with open(self.experiments_file, "w") as fp:
            json.dump(exp_f, fp)

        if os.path.exists(prefix.joinpath(f"{exp_name}_grads.mmap")):
            mode = "r+"
        else:
            mode = "w+"
        self.load_current_store(
            model_id=model_id, exp_name=exp_name, exp_num_targets=num_targets, mode=mode
        )

    def _load(self, fname, shape, mode, dtype=None):
        if mode == "w+":
            self.logger.debug(f"Creating {fname}.")
        else:
            self.logger.debug(f"Loading {fname}.")
        if dtype is None:
            dtype = np.float16 if self.use_half_precision else np.float32
        try:
            return open_memmap(filename=fname, mode=mode, shape=shape, dtype=dtype)
        except OSError:
            self.logger.info(f"{fname} does not exist, skipping.")
            return None

    def load_current_store(
        self,
        model_id: int,
        exp_name: Optional[str] = None,
        exp_num_targets: Optional[int] = -1,
        mode: Optional[str] = "r+",
    ) -> None:
        """This method uses numpy memmaps for serializing the TRAK results and
        intermediate values.

        Args:
            model_id (int):
                a unique ID for a checkpoint
            exp_name (str, optional):
                Experiment name for which to load the features. If None, loads
                the train (source) features for a model ID. Defaults to None.
            exp_num_targets (int, optional):
                Number of targets for the experiment. Specify only when exp_name
                is not None. Defaults to -1.
            mode (str, optional):
                Defaults to 'r+'.

        """
        self.current_model_id = model_id
        if exp_name is not None:
            self.current_experiment_name = exp_name
        prefix = self.save_dir.joinpath(str(self.current_model_id))

        if exp_name is None:
            to_load = {
                "grads": (
                    prefix.joinpath("grads.mmap"),
                    (self.train_set_size, self.proj_dim),
                    None,
                ),
                "out_to_loss": (
                    prefix.joinpath("out_to_loss.mmap"),
                    (self.train_set_size, 1),
                    None,
                ),
                "features": (
                    prefix.joinpath("features.mmap"),
                    (self.train_set_size, self.proj_dim),
                    None,
                ),
                "is_featurized": (
                    prefix.joinpath("_is_featurized.mmap"),
                    (self.train_set_size, 1),
                    np.int32,
                ),
            }
        else:
            to_load = {
                f"{exp_name}_grads": (
                    prefix.joinpath(f"{exp_name}_grads.mmap"),
                    (exp_num_targets, self.proj_dim),
                    None,
                ),
                f"{exp_name}_scores": (
                    self.save_dir.joinpath(f"scores/{exp_name}.mmap"),
                    (self.train_set_size, exp_num_targets),
                    None,
                ),
            }

        for name, (path, shape, dtype) in to_load.items():
            self.current_store[name] = self._load(path, shape, mode, dtype)

    def save_scores(self, exp_name):
        assert self.current_experiment_name == exp_name
        prefix = self.save_dir.joinpath("scores")
        self.logger.info(f"Saving scores in {prefix}/{exp_name}.mmap")
        self.current_store[f"{exp_name}_scores"].flush()
        self.experiments[exp_name]["scores_finalized"] = 1
        with open(self.experiments_file, "w") as fp:
            json.dump(self.experiments, fp)

    def del_grads(self, model_id):
        grads_file = self.save_dir.joinpath(str(model_id)).joinpath("grads.mmap")

        # delete grads memmap
        grads_file.unlink()
