# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy
import time
import logging

from libc.stdint cimport uint8_t, int32_t, int64_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool as c_bool
from libcpp.cast cimport const_cast
from libcpp.memory cimport (
    dynamic_pointer_cast,
    make_shared,
    shared_ptr,
    unique_ptr,
)
from libcpp.string cimport string as c_string
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector as c_vector

from cython.operator import dereference, postincrement

from ray.includes.common cimport (
    CLanguage,
    CRayObject,
    CRayStatus,
    CGcsClientOptions,
    CTaskArg,
    CRayFunction,
    LocalMemoryBuffer,
    move,
    LANGUAGE_CPP,
    LANGUAGE_JAVA,
    LANGUAGE_PYTHON,
    WORKER_TYPE_WORKER,
    WORKER_TYPE_DRIVER,
)
from ray.includes.libraylet cimport (
    CRayletClient,
    GCSProfileEvent,
    GCSProfileTableData,
    WaitResultPair,
)
from ray.includes.unique_ids cimport (
    CActorID,
    CActorCheckpointID,
    CObjectID,
    CClientID,
)
import ray
import ray.experimental.signal as ray_signal
from ray import profiling
from ray.includes.libcoreworker cimport (
    CCoreWorker, CTaskOptions, ResourceMappingType)
from ray.includes.task cimport CTaskSpec
from ray.includes.ray_config cimport RayConfig
from ray.exceptions import RayletError, ObjectStoreFullError
from ray.utils import decode
from ray.ray_constants import (
    DEFAULT_PUT_OBJECT_DELAY,
    DEFAULT_PUT_OBJECT_RETRIES,
    RAW_BUFFER_METADATA,
    PICKLE5_BUFFER_METADATA,
)

# pyarrow cannot be imported until after _raylet finishes initializing
# (see ray/__init__.py for details).
# Unfortunately, Cython won't compile if 'pyarrow' is undefined, so we
# "forward declare" it here and then replace it with a reference to the
# imported package from ray/__init__.py.
# TODO(edoakes): Fix this.
pyarrow = None

cimport cpython

include "includes/unique_ids.pxi"
include "includes/ray_config.pxi"
include "includes/task.pxi"
include "includes/buffer.pxi"
include "includes/common.pxi"
include "includes/serialization.pxi"
include "includes/libcoreworker.pxi"


logger = logging.getLogger(__name__)


if cpython.PY_MAJOR_VERSION >= 3:
    import pickle
else:
    import cPickle as pickle


cdef int check_status(const CRayStatus& status) nogil except -1:
    if status.ok():
        return 0

    with gil:
        message = status.message().decode()

    if status.IsObjectStoreFull():
        raise ObjectStoreFullError(message)
    else:
        raise RayletError(message)


cdef VectorToObjectIDs(const c_vector[CObjectID] &object_ids):
    result = []
    for i in range(object_ids.size()):
        result.append(ObjectID(object_ids[i].Binary()))
    return result


cdef c_vector[CObjectID] ObjectIDsToVector(object_ids):
    """A helper function that converts a Python list of object IDs to a vector.

    Args:
        object_ids (list): The Python list of object IDs.

    Returns:
        The output vector.
    """
    cdef:
        ObjectID object_id
        c_vector[CObjectID] result
    for object_id in object_ids:
        result.push_back(object_id.native())
    return result


def compute_put_id(TaskID task_id, int64_t put_index):
    if put_index < 1 or put_index > <int64_t>CObjectID.MaxObjectIndex():
        raise ValueError("The range of 'put_index' should be [1, %d]"
                         % CObjectID.MaxObjectIndex())
    return ObjectID(CObjectID.ForPut(task_id.native(), put_index, 0).Binary())


def compute_task_id(ObjectID object_id):
    return TaskID(object_id.native().TaskId().Binary())


cdef c_bool is_simple_value(value, int64_t *num_elements_contained):
    num_elements_contained[0] += 1

    if num_elements_contained[0] >= RayConfig.instance().num_elements_limit():
        return False

    if (cpython.PyInt_Check(value) or cpython.PyLong_Check(value) or
            value is False or value is True or cpython.PyFloat_Check(value) or
            value is None):
        return True

    if cpython.PyBytes_CheckExact(value):
        num_elements_contained[0] += cpython.PyBytes_Size(value)
        return (num_elements_contained[0] <
                RayConfig.instance().num_elements_limit())

    if cpython.PyUnicode_CheckExact(value):
        num_elements_contained[0] += cpython.PyUnicode_GET_SIZE(value)
        return (num_elements_contained[0] <
                RayConfig.instance().num_elements_limit())

    if (cpython.PyList_CheckExact(value) and
            cpython.PyList_Size(value) < RayConfig.instance().size_limit()):
        for item in value:
            if not is_simple_value(item, num_elements_contained):
                return False
        return (num_elements_contained[0] <
                RayConfig.instance().num_elements_limit())

    if (cpython.PyDict_CheckExact(value) and
            cpython.PyDict_Size(value) < RayConfig.instance().size_limit()):
        # TODO(suquark): Using "items" in Python2 is not very efficient.
        for k, v in value.items():
            if not (is_simple_value(k, num_elements_contained) and
                    is_simple_value(v, num_elements_contained)):
                return False
        return (num_elements_contained[0] <
                RayConfig.instance().num_elements_limit())

    if (cpython.PyTuple_CheckExact(value) and
            cpython.PyTuple_Size(value) < RayConfig.instance().size_limit()):
        for item in value:
            if not is_simple_value(item, num_elements_contained):
                return False
        return (num_elements_contained[0] <
                RayConfig.instance().num_elements_limit())

    if isinstance(value, numpy.ndarray):
        if value.dtype == "O":
            return False
        num_elements_contained[0] += value.nbytes
        return (num_elements_contained[0] <
                RayConfig.instance().num_elements_limit())

    return False


def check_simple_value(value):
    """Check if value is simple enough to be send by value.

    This method checks if a Python object is sufficiently simple that it can
    be serialized and passed by value as an argument to a task (without being
    put in the object store). The details of which objects are sufficiently
    simple are defined by this method and are not particularly important.
    But for performance reasons, it is better to place "small" objects in
    the task itself and "large" objects in the object store.

    Args:
        value: Python object that should be checked.

    Returns:
        True if the value should be send by value, False otherwise.
    """

    cdef int64_t num_elements_contained = 0
    return is_simple_value(value, &num_elements_contained)


cdef class Language:
    cdef CLanguage lang

    def __cinit__(self, int32_t lang):
        self.lang = <CLanguage>lang

    @staticmethod
    cdef from_native(const CLanguage& lang):
        return Language(<int32_t>lang)

    def __eq__(self, other):
        return (isinstance(other, Language) and
                (<int32_t>self.lang) == (<int32_t>other.lang))

    def __repr__(self):
        if <int32_t>self.lang == <int32_t>LANGUAGE_PYTHON:
            return "PYTHON"
        elif <int32_t>self.lang == <int32_t>LANGUAGE_CPP:
            return "CPP"
        elif <int32_t>self.lang == <int32_t>LANGUAGE_JAVA:
            return "JAVA"
        else:
            raise Exception("Unexpected error")


# Programming language enum values.
cdef Language LANG_PYTHON = Language.from_native(LANGUAGE_PYTHON)
cdef Language LANG_CPP = Language.from_native(LANGUAGE_CPP)
cdef Language LANG_JAVA = Language.from_native(LANGUAGE_JAVA)


cdef unordered_map[c_string, double] resource_map_from_dict(resource_map):
    cdef:
        unordered_map[c_string, double] out
        c_string resource_name
    if not isinstance(resource_map, dict):
        raise TypeError("resource_map must be a dictionary")
    for key, value in resource_map.items():
        out[key.encode("ascii")] = float(value)
    return out


# Required because the Cython parser rejects 'unordered_map[c_string, double]*'
# as a template parameter.
ctypedef unordered_map[c_string, double]* ResourceMapPtr

cdef dict resource_map_to_dict(const unordered_map[c_string, double] &map):
    result = {}
    # This const_cast is required because Cython doesn't properly support
    # const_iterators.
    for key, value in dereference(const_cast[ResourceMapPtr](&map)):
        result[key.decode("ascii")] = value

    return result


cdef c_vector[c_string] string_vector_from_list(list string_list):
    cdef:
        c_vector[c_string] out
    for s in string_list:
        if not isinstance(s, bytes):
            raise TypeError("string_list elements must be bytes")
        out.push_back(s)
    return out


cdef class RayletClient:
    cdef CRayletClient* client

    def __cinit__(self, CoreWorker core_worker):
        # The core worker and raylet client need to share an underlying
        # raylet client, so we take a reference to the core worker's client
        # here. The client is a raw pointer because it is only a temporary
        # workaround and will be removed once the core worker transition is
        # complete, so we don't want to change the unique_ptr in core worker
        # to a shared_ptr. This means the core worker *must* be
        # initialized before the raylet client.
        self.client = &core_worker.core_worker.get().GetRayletClient()

    def submit_task(self, TaskSpec task_spec):
        cdef:
            CObjectID c_id

        check_status(self.client.SubmitTask(
            task_spec.task_spec[0]))

    def fetch_or_reconstruct(self, object_ids,
                             c_bool fetch_only,
                             TaskID current_task_id=TaskID.nil()):
        cdef c_vector[CObjectID] fetch_ids = ObjectIDsToVector(object_ids)
        check_status(self.client.FetchOrReconstruct(
            fetch_ids, fetch_only, current_task_id.native()))

    def push_error(self, JobID job_id, error_type, error_message,
                   double timestamp):
        check_status(self.client.PushError(job_id.native(),
                                           error_type.encode("ascii"),
                                           error_message.encode("ascii"),
                                           timestamp))

    def prepare_actor_checkpoint(self, ActorID actor_id):
        cdef:
            CActorCheckpointID checkpoint_id
            CActorID c_actor_id = actor_id.native()

        # PrepareActorCheckpoint will wait for raylet's reply, release
        # the GIL so other Python threads can run.
        with nogil:
            check_status(self.client.PrepareActorCheckpoint(
                c_actor_id, checkpoint_id))
        return ActorCheckpointID(checkpoint_id.Binary())

    def notify_actor_resumed_from_checkpoint(self, ActorID actor_id,
                                             ActorCheckpointID checkpoint_id):
        check_status(self.client.NotifyActorResumedFromCheckpoint(
            actor_id.native(), checkpoint_id.native()))

    def set_resource(self, basestring resource_name,
                     double capacity, ClientID client_id):
        self.client.SetResource(resource_name.encode("ascii"), capacity,
                                CClientID.FromBinary(client_id.binary()))

    @property
    def job_id(self):
        return JobID(self.client.GetJobID().Binary())

    @property
    def is_worker(self):
        return self.client.IsWorker()

cdef list deserialize_args(const c_vector[CTaskArg] &args):
    with profiling.profile("task:deserialize_arguments"):
        result = []
        for i in range(args.size()):
            if args[i].IsPassedByReference():
                result.append(
                    ray.get(ObjectID(args[i].GetReference().Binary())))
            else:
                data = Buffer.make(args[i].GetValue().GetData())
                if (args[i].GetValue().HasMetadata()
                    and Buffer.make(
                        args[i].GetValue().GetMetadata()).to_pybytes()
                        == RAW_BUFFER_METADATA):
                    result.append(data)
                else:
                    result.append(pickle.loads(data))

    return result

cdef CRayStatus execute_normal_task(
        const CRayFunction &ray_function,
        const CJobID &c_job_id, const CTaskID &c_task_id,
        const c_vector[CTaskArg] &args,
        const c_vector[CObjectID] &return_ids,
        c_vector[shared_ptr[CRayObject]] *returns) nogil:

    with gil:
        worker = ray.worker.global_worker

        # Automatically restrict the GPUs available to this task.
        ray.utils.set_cuda_visible_devices(ray.get_gpu_ids())

        job_id = JobID(c_job_id.Binary())
        task_id = TaskID(c_task_id.Binary())

        assert worker.current_task_id.is_nil()
        assert worker.task_context.task_index == 0
        assert worker.task_context.put_index == 1
        # If this worker is not an actor, check that `current_job_id`
        # was reset when the worker finished the previous task.
        assert worker.current_job_id.is_nil()
        # Set the driver ID of the current running task. This is
        # needed so that if the task throws an exception, we propagate
        # the error message to the correct driver.
        worker.current_job_id = job_id
        worker.core_worker.set_current_job_id(job_id)
        worker.task_context.current_task_id = task_id
        worker.core_worker.set_current_task_id(task_id)

        worker.current_job_id = job_id
        worker._process_normal_task(
            ray_function.GetFunctionDescriptor(),
            job_id,
            task_id,
            deserialize_args(args),
            VectorToObjectIDs(return_ids),
        )

        # Reset the state fields so the next task can run.
        worker.task_context.current_task_id = TaskID.nil()
        worker.core_worker.set_current_task_id(TaskID.nil())
        worker.task_context.task_index = 0
        worker.task_context.put_index = 1
        # Don't need to reset `current_job_id` if the worker is an
        # actor. Because the following tasks should all have the
        # same driver id.
        worker.current_job_id = WorkerID.nil()
        worker.core_worker.set_current_job_id(JobID.nil())
        # Reset signal counters so that the next task can get
        # all past signals.
        ray_signal.reset()

    return CRayStatus.OK()

cdef CRayStatus execute_actor_task(
        const CRayFunction &ray_function,
        const CJobID &c_job_id, const CTaskID &c_task_id,
        const CActorID &c_actor_id, c_bool create_actor,
        const unordered_map[c_string, double] &resources,
        const c_vector[CTaskArg] &args,
        const c_vector[CObjectID] &return_ids,
        c_vector[shared_ptr[CRayObject]] *returns) nogil:

    with gil:
        worker = ray.worker.global_worker

        # Automatically restrict the GPUs available to this task.
        ray.utils.set_cuda_visible_devices(ray.get_gpu_ids())

        job_id = JobID(c_job_id.Binary())
        task_id = TaskID(c_task_id.Binary())

        assert worker.current_task_id.is_nil()
        assert worker.task_context.task_index == 0
        assert worker.task_context.put_index == 1
        if create_actor:
            # If this worker is not an actor, check that
            # `current_job_id` was reset when the worker finished the
            # previous task.
            assert worker.current_job_id.is_nil()
            # Set the driver ID of the current running task. This is
            # needed so that if the task throws an exception, we
            # propagate the error message to the correct driver.
            worker.current_job_id = job_id
            worker.core_worker.set_current_job_id(job_id)
        else:
            # If this worker is an actor, current_job_id wasn't reset.
            # Check that current task's driver ID equals the previous
            # one.
            assert worker.current_job_id == job_id

        worker.task_context.current_task_id = task_id
        worker.core_worker.set_current_task_id(task_id)

        worker._process_actor_task(
            ray_function.GetFunctionDescriptor(),
            job_id,
            task_id,
            ActorID(c_actor_id.Binary()),
            resource_map_to_dict(resources),
            deserialize_args(args),
            VectorToObjectIDs(return_ids),
            create_actor,
        )

        # Reset the state fields so the next task can run.
        worker.task_context.current_task_id = TaskID.nil()
        worker.core_worker.set_current_task_id(TaskID.nil())
        worker.task_context.task_index = 0
        worker.task_context.put_index = 1

    return CRayStatus.OK()

cdef class CoreWorker:
    cdef unique_ptr[CCoreWorker] core_worker

    def __cinit__(self, is_driver, store_socket, raylet_socket,
                  JobID job_id, GcsClientOptions gcs_options, log_dir,
                  node_ip_address):
        assert pyarrow is not None, ("Expected pyarrow to be imported from "
                                     "outside _raylet. See __init__.py for "
                                     "details.")

        self.core_worker.reset(new CCoreWorker(
            WORKER_TYPE_DRIVER if is_driver else WORKER_TYPE_WORKER,
            LANGUAGE_PYTHON, store_socket.encode("ascii"),
            raylet_socket.encode("ascii"), job_id.native(),
            gcs_options.native()[0], log_dir.encode("utf-8"),
            node_ip_address.encode("utf-8"), execute_normal_task,
            execute_actor_task, False))

    def disconnect(self):
        with nogil:
            self.core_worker.get().Disconnect()

    def run_task_loop(self):
        self.core_worker.get().Execution().Run()

    def get_objects(self, object_ids, TaskID current_task_id):
        cdef:
            c_vector[shared_ptr[CRayObject]] results
            CTaskID c_task_id = current_task_id.native()
            c_vector[CObjectID] c_object_ids = ObjectIDsToVector(object_ids)

        with nogil:
            check_status(self.core_worker.get().Objects().Get(
                c_object_ids, -1, &results))

        data_metadata_pairs = []
        for result in results:
            # core_worker will return a nullptr for objects that couldn't be
            # retrieved from the store or if an object was an exception.
            if not result.get():
                data_metadata_pairs.append((None, None))
            else:
                data = None
                metadata = None
                if result.get().HasData():
                    data = Buffer.make(result.get().GetData())
                if result.get().HasMetadata():
                    metadata = Buffer.make(
                        result.get().GetMetadata()).to_pybytes()
                data_metadata_pairs.append((data, metadata))

        return data_metadata_pairs

    def object_exists(self, ObjectID object_id):
        cdef:
            c_bool has_object
            CObjectID c_object_id = object_id.native()

        with nogil:
            check_status(self.core_worker.get().Objects().Contains(
                c_object_id, &has_object))

        return has_object

    def put_serialized_object(self, serialized_object, ObjectID object_id,
                              int memcopy_threads=6, c_bool put_async=False):
        cdef:
            shared_ptr[CBuffer] data
            shared_ptr[CBuffer] metadata
            shared_ptr[CRayObject] ray_object
            CObjectID c_object_id = object_id.native()
            int64_t data_size

        data_size = serialized_object.total_bytes

        # If we are returning values from a task, and the return objects are
        # small, we can batch the puts until task batch completion. This saves
        # a lot of IPCs to plasma for small objects.
        if put_async and data_size < RayConfig.instance().size_limit():
            data = dynamic_pointer_cast[
                CBuffer, LocalMemoryBuffer](
                    make_shared[LocalMemoryBuffer](
                        <uint8_t*>NULL, data_size, True))
            stream = pyarrow.FixedSizeBufferWriter(
                pyarrow.py_buffer(Buffer.make(data)))
            serialized_object.write_to(stream)
            ray_object = make_shared[CRayObject](data, metadata)
            with nogil:
                check_status(self.core_worker.get().Objects().PutAsync(
                    dereference(ray_object), c_object_id))
            return  # Done, async puts will be flushed with task batch.

        # Non-async path.
        with nogil:
            check_status(self.core_worker.get().Objects().Create(
                        metadata, data_size, c_object_id, &data))

        # If data is nullptr, that means the ObjectID already existed,
        # which we ignore.
        # TODO(edoakes): this is hacky, we should return the error instead
        # and deal with it here.
        if not data:
            return

        stream = pyarrow.FixedSizeBufferWriter(
            pyarrow.py_buffer(Buffer.make(data)))
        stream.set_memcopy_threads(memcopy_threads)
        serialized_object.write_to(stream)

        with nogil:
            check_status(self.core_worker.get().Objects().Seal(c_object_id))

    def put_raw_buffer(self, c_string value, ObjectID object_id,
                       int memcopy_threads=6):
        cdef:
            c_string metadata_str = RAW_BUFFER_METADATA
            CObjectID c_object_id = object_id.native()
            shared_ptr[CBuffer] data
            shared_ptr[CBuffer] metadata = dynamic_pointer_cast[
                CBuffer, LocalMemoryBuffer](
                    make_shared[LocalMemoryBuffer](
                        <uint8_t*>(metadata_str.data()), metadata_str.size()))

        with nogil:
            check_status(self.core_worker.get().Objects().Create(
                metadata, value.size(), c_object_id, &data))

        stream = pyarrow.FixedSizeBufferWriter(
            pyarrow.py_buffer(Buffer.make(data)))
        stream.set_memcopy_threads(memcopy_threads)
        stream.write(pyarrow.py_buffer(value))

        with nogil:
            check_status(self.core_worker.get().Objects().Seal(c_object_id))

    def put_pickle5_buffers(self, ObjectID object_id, c_string inband,
                            Pickle5Writer writer,
                            int memcopy_threads):
        cdef:
            shared_ptr[CBuffer] data
            c_string metadata_str = PICKLE5_BUFFER_METADATA
            shared_ptr[CBuffer] metadata = dynamic_pointer_cast[
                CBuffer, LocalMemoryBuffer](
                    make_shared[LocalMemoryBuffer](
                        <uint8_t*>(metadata_str.data()), metadata_str.size()))
            CObjectID c_object_id = object_id.native()
            size_t data_size

        data_size = writer.get_total_bytes(inband)

        with nogil:
            check_status(self.core_worker.get().Objects().Create(
                        metadata, data_size, c_object_id, &data))

        # If data is nullptr, that means the ObjectID already existed,
        # which we ignore.
        # TODO(edoakes): this is hacky, we should return the error instead
        # and deal with it here.
        if not data:
            return

        writer.write_to(inband, data, memcopy_threads)
        with nogil:
            check_status(self.core_worker.get().Objects().Seal(c_object_id))

    def wait(self, object_ids, int num_returns, int64_t timeout_milliseconds,
             TaskID current_task_id):
        cdef:
            WaitResultPair result
            c_vector[CObjectID] wait_ids
            c_vector[c_bool] results
            CTaskID c_task_id = current_task_id.native()

        wait_ids = ObjectIDsToVector(object_ids)
        with nogil:
            check_status(self.core_worker.get().Objects().Wait(
                wait_ids, num_returns, timeout_milliseconds, &results))

        assert len(results) == len(object_ids)

        ready, not_ready = [], []
        for i, object_id in enumerate(object_ids):
            if results[i]:
                ready.append(object_id)
            else:
                not_ready.append(object_id)

        return (ready, not_ready)

    def free_objects(self, object_ids, c_bool local_only,
                     c_bool delete_creating_tasks):
        cdef:
            c_vector[CObjectID] free_ids = ObjectIDsToVector(object_ids)

        with nogil:
            check_status(self.core_worker.get().Objects().Delete(
                free_ids, local_only, delete_creating_tasks))

    def get_current_task_id(self):
        return TaskID(self.core_worker.get().GetCurrentTaskId().Binary())

    def set_current_task_id(self, TaskID task_id):
        cdef:
            CTaskID c_task_id = task_id.native()

        with nogil:
            self.core_worker.get().SetCurrentTaskId(c_task_id)

    def set_current_job_id(self, JobID job_id):
        cdef:
            CJobID c_job_id = job_id.native()

        with nogil:
            self.core_worker.get().SetCurrentJobId(c_job_id)

    def submit_task(self,
                    function_descriptor,
                    args,
                    int num_return_vals,
                    resources):
        cdef:
            unordered_map[c_string, double] c_resources
            CTaskOptions task_options
            CRayFunction ray_function
            c_vector[CTaskArg] args_vector
            c_vector[CObjectID] return_ids
            c_string pickled_str
            shared_ptr[CBuffer] arg_data
            shared_ptr[CBuffer] arg_metadata

        c_resources = resource_map_from_dict(resources)
        task_options = CTaskOptions(num_return_vals, c_resources)
        ray_function = CRayFunction(
            LANGUAGE_PYTHON, string_vector_from_list(function_descriptor))

        for arg in args:
            if isinstance(arg, ObjectID):
                args_vector.push_back(
                    CTaskArg.PassByReference((<ObjectID>arg).native()))
            else:
                pickled_str = pickle.dumps(
                    arg, protocol=pickle.HIGHEST_PROTOCOL)
                arg_data = dynamic_pointer_cast[CBuffer, LocalMemoryBuffer](
                        make_shared[LocalMemoryBuffer](
                            <uint8_t*>(pickled_str.data()),
                            pickled_str.size(),
                            True))
                args_vector.push_back(
                    CTaskArg.PassByValue(
                        make_shared[CRayObject](arg_data, arg_metadata)))

        with nogil:
            check_status(self.core_worker.get().Tasks().SubmitTask(
                ray_function, args_vector, task_options, &return_ids))

        return VectorToObjectIDs(return_ids)

    def set_object_store_client_options(self, c_string client_name,
                                        int64_t limit_bytes):
        with nogil:
            check_status(self.core_worker.get().Objects().SetClientOptions(
                client_name, limit_bytes))

    def object_store_memory_usage_string(self):
        cdef:
            c_string message

        with nogil:
            message = self.core_worker.get().Objects().MemoryUsageString()

        return message.decode("utf-8")

    def resource_ids(self):
        cdef:
            ResourceMappingType resource_mapping = (
                self.core_worker.get().Execution().GetResourceIDs())
            unordered_map[
                c_string, c_vector[pair[int64_t, double]]
            ].iterator iterator = resource_mapping.begin()
            c_vector[pair[int64_t, double]] c_value

        resources_dict = {}
        while iterator != resource_mapping.end():
            key = decode(dereference(iterator).first)
            c_value = dereference(iterator).second
            ids_and_fractions = []
            for i in range(c_value.size()):
                ids_and_fractions.append(
                    (c_value[i].first, c_value[i].second))
            resources_dict[key] = ids_and_fractions
            postincrement(iterator)

        return resources_dict

    def profile_event(self, event_type, dict extra_data):
        cdef:
            c_string c_event_type = event_type.encode("ascii")

        return ProfileEvent.make(
            self.core_worker.get().CreateProfileEvent(c_event_type),
            extra_data)
