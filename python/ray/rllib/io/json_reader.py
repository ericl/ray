from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import logging
import os
import random
from six.moves.urllib.parse import urlparse

try:
    from smart_open import smart_open
except ImportError:
    smart_open = None

from ray.rllib.io.input_reader import InputReader
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.utils.compression import unpack_if_needed

logger = logging.getLogger(__name__)


class JsonReader(InputReader):
    """Reader object that loads experiences from JSON file chunks.

    The input files will be read from in an random order."""

    def __init__(self, ioctx, inputs):
        """Initialize a JsonReader.

        Arguments:
            ioctx (IOContext): current IO context object.
            inputs (str|list): either a glob expression for files, e.g.,
                "/tmp/**/*.json", or a list of single file paths or URIs, e.g.,
                ["s3://bucket/file.json", "s3://bucket/file2.json"].
        """

        self.ioctx = ioctx
        if type(inputs) is str:
            if os.path.isdir(inputs):
                inputs = os.path.join(inputs, "*.json")
                logger.warn(
                    "Treating input directory as glob pattern: {}".format(
                        inputs))
            if urlparse(inputs).scheme:
                raise ValueError(
                    "Don't know how to glob over `{}`, ".format(inputs) +
                    "please specify a list of files to read instead.")
            else:
                self.files = glob.glob(inputs)
        elif type(inputs) is list:
            self.files = inputs
        else:
            raise ValueError(
                "type of inputs must be list or str, not {}".format(inputs))
        if self.files:
            logger.info("Found {} input files.".format(len(self.files)))
        else:
            raise ValueError("No files found matching {}".format(inputs))
        self.cur_file = None

    def next(self):
        batch = self._try_parse(self._next_line())
        tries = 0
        while not batch and tries < 100:
            tries += 1
            logger.debug("Skipping empty line in {}".format(self.cur_file))
            batch = self._try_parse(self._next_line())
        if not batch:
            raise ValueError(
                "Failed to read valid experience batch from file: {}".format(
                    self.cur_file))
        return batch

    def _try_parse(self, line):
        line = line.strip()
        if not line:
            return None
        try:
            return _from_json(line)
        except Exception as e:
            logger.warn("Ignoring corrupt json record in {}: {}: {}".format(
                self.cur_file, line, e))
            return None

    def _next_line(self):
        if not self.cur_file:
            self.cur_file = self._next_file()
        line = self.cur_file.readline()
        tries = 0
        while not line and tries < 100:
            tries += 1
            if hasattr(self.cur_file, "close"):  # legacy smart_open impls
                self.cur_file.close()
            self.cur_file = self._next_file()
            line = self.cur_file.readline()
            if not line:
                logger.debug("Ignoring empty file {}".format(self.cur_file))
        if not line:
            raise ValueError("Failed to read next line from files: {}".format(
                self.files))
        return line

    def _next_file(self):
        path = random.choice(self.files)
        if urlparse(path).scheme:
            if smart_open is None:
                raise ValueError(
                    "You must install the `smart_open` module to read "
                    "from URIs like {}".format(path))
            return smart_open(path, "r")
        else:
            return open(path, "r")


def _from_json(batch):
    if isinstance(batch, bytes):  # smart_open S3 doesn't respect "r"
        batch = batch.decode("utf-8")
    data = json.loads(batch)
    for k, v in data.items():
        data[k] = [unpack_if_needed(x) for x in unpack_if_needed(v)]
    return SampleBatch(data)
