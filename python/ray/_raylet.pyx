# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy
import time
import logging

from libc.stdint cimport uint8_t, int32_t, int64_t, uint64_t
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
    CTaskType,
    CRayFunction,
    LocalMemoryBuffer,
    move,
    LANGUAGE_CPP,
    LANGUAGE_JAVA,
    LANGUAGE_PYTHON,
    LocalMemoryBuffer,
    TASK_TYPE_NORMAL_TASK,
    TASK_TYPE_ACTOR_CREATION_TASK,
    TASK_TYPE_ACTOR_TASK,
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
from ray.includes.libcoreworker cimport (
    CActorCreationOptions,
    CCoreWorker,
    CTaskOptions,
    ResourceMappingType,
)
from ray.includes.task cimport CTaskSpec
from ray.includes.ray_config cimport RayConfig

import ray
import ray.experimental.signal as ray_signal
from ray import profiling
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

cdef RayObjectsToDataMetadataPairs(
        const c_vector[shared_ptr[CRayObject]] objects):
    data_metadata_pairs = []
    for i in range(objects.size()):
        # core_worker will return a nullptr for objects that couldn't be
        # retrieved from the store or if an object was an exception.
        if not objects[i].get():
            data_metadata_pairs.append((None, None))
        else:
            data = None
            metadata = None
            if objects[i].get().HasData():
                data = Buffer.make(objects[i].get().GetData())
            if objects[i].get().HasMetadata():
                metadata = Buffer.make(
                    objects[i].get().GetMetadata()).to_pybytes()
            data_metadata_pairs.append((data, metadata))
    return data_metadata_pairs


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


cdef int prepare_resources(
        dict resource_dict,
        unordered_map[c_string, double] *resource_map) except -1:
    cdef:
        unordered_map[c_string, double] out
        c_string resource_name

    if resource_dict is None:
        raise ValueError("Must provide resource map.")

    for key, value in resource_dict.items():
        if not (isinstance(value, int) or isinstance(value, float)):
            raise ValueError("Resource quantities may only be ints or floats.")
        if value < 0:
            raise ValueError("Resource quantities may not be negative.")
        if value > 0:
            if (value >= 1 and isinstance(value, float)
                    and not value.is_integer()):
                raise ValueError(
                    "Resource quantities >1 must be whole numbers.")
            resource_map[0][key.encode("ascii")] = float(value)
    return 0


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


cdef void prepare_args(list args, c_vector[CTaskArg] *args_vector):
    cdef:
        c_string pickled_str
        shared_ptr[CBuffer] arg_data
        shared_ptr[CBuffer] arg_metadata

    for arg in args:
        if isinstance(arg, ObjectID):
            args_vector.push_back(
                CTaskArg.PassByReference((<ObjectID>arg).native()))
        elif not ray._raylet.check_simple_value(arg):
            args_vector.push_back(
                CTaskArg.PassByReference((<ObjectID>ray.put(arg)).native()))
        else:
            pickled_str = pickle.dumps(
                arg, protocol=pickle.HIGHEST_PROTOCOL)
            # TODO(edoakes): This makes a copy that could be avoided.
            arg_data = dynamic_pointer_cast[CBuffer, LocalMemoryBuffer](
                    make_shared[LocalMemoryBuffer](
                        <uint8_t*>(pickled_str.data()),
                        pickled_str.size(),
                        True))
            args_vector.push_back(
                CTaskArg.PassByValue(
                    make_shared[CRayObject](arg_data, arg_metadata)))


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

cdef list deserialize_args(
        const c_vector[shared_ptr[CRayObject]] &args,
        const c_vector[CObjectID] &arg_reference_ids):
    cdef:
        c_vector[shared_ptr[CRayObject]] by_reference_objects

    with profiling.profile("task:deserialize_arguments"):
        results = []
        by_reference_ids = []
        by_reference_indices = []
        for i in range(args.size()):
            # Passed by value.
            if arg_reference_ids[i].IsNil():
                data = Buffer.make(args[i].get().GetData())
                if (args[i].get().HasMetadata()
                    and Buffer.make(
                        args[i].get().GetMetadata()).to_pybytes()
                        == RAW_BUFFER_METADATA):
                    results.append(data)
                else:
                    results.append(pickle.loads(data.to_pybytes()))
            # Passed by reference.
            else:
                by_reference_ids.append(
                    ObjectID(arg_reference_ids[i].Binary()))
                by_reference_indices.append(i)
                by_reference_objects.push_back(args[i])
                results.append(None)

        data_metadata_pairs = RayObjectsToDataMetadataPairs(
            by_reference_objects)
        for i, result in enumerate(
            ray.worker.global_worker.deserialize_objects(
                data_metadata_pairs, by_reference_ids)):
            results[by_reference_indices[i]] = result

    return results

cdef CRayStatus execute_task(
        CTaskType task_type,
        const CRayFunction &ray_function,
        const CActorID &c_actor_id,
        const unordered_map[c_string, double] &resources,
        const c_vector[shared_ptr[CRayObject]] &args,
        const c_vector[CObjectID] &arg_reference_ids,
        const c_vector[CObjectID] &return_ids,
        c_bool is_direct_call,
        c_vector[shared_ptr[CRayObject]] *returns) nogil:

    with gil:
        worker = ray.worker.global_worker

        # Automatically restrict the GPUs available to this task.
        ray.utils.set_cuda_visible_devices(ray.get_gpu_ids())

        # TODO(ekl) this is set by the actor constructor
        # assert worker.current_task_id.is_nil()
        assert worker.task_context.task_index == 0
        assert worker.task_context.put_index == 1

        job_id = worker.core_worker.get_current_job_id()
        task_id = worker.core_worker.get_current_task_id()

        # If this worker is not an actor, check that `current_job_id`
        # was reset when the worker finished the previous task.
        if <int>task_type in [<int>TASK_TYPE_NORMAL_TASK,
                              <int>TASK_TYPE_ACTOR_CREATION_TASK]:
            assert worker.current_job_id.is_nil()
            # Set the driver ID of the current running task. This is
            # needed so that if the task throws an exception, we propagate
            # the error message to the correct driver.
            worker.current_job_id = job_id
        else:
            # If this worker is an actor, current_job_id wasn't reset.
            # Check that current task's driver ID equals the previous
            # one.
            assert worker.current_job_id == job_id

        worker.task_context.current_task_id = task_id

        if <int>task_type == <int>TASK_TYPE_NORMAL_TASK:
            worker._process_normal_task(
                ray_function.GetFunctionDescriptor(),
                job_id,
                task_id,
                deserialize_args(args, arg_reference_ids),
                VectorToObjectIDs(return_ids),
            )
        else:
            py_returns = worker._process_actor_task(
                ray_function.GetFunctionDescriptor(),
                job_id,
                task_id,
                ActorID(c_actor_id.Binary()),
                resource_map_to_dict(resources),
                deserialize_args(args, arg_reference_ids),
                VectorToObjectIDs(return_ids),
                <int>task_type == <int>TASK_TYPE_ACTOR_CREATION_TASK,
                is_direct_call,
            )

            if is_direct_call:
                to_ray_objects(py_returns, returns)  # TODO(ekl)

        # Reset the state fields so the next task can run.
        worker.task_context.current_task_id = TaskID.nil()
        worker.core_worker.set_current_task_id(TaskID.nil())
        worker.task_context.task_index = 0
        worker.task_context.put_index = 1

        # Don't need to reset `current_job_id` if the worker is an
        # actor. Because the following tasks should all have the
        # same driver id.
        if <int>task_type == <int>TASK_TYPE_NORMAL_TASK:
            worker.current_job_id = WorkerID.nil()
            worker.core_worker.set_current_job_id(JobID.nil())

        # Reset signal counters so that the next task can get
        # all past signals.
        ray_signal.reset()

    return CRayStatus.OK()

cdef void to_ray_objects(
        py_objects,
        c_vector[shared_ptr[CRayObject]] *returns) nogil:

    cdef:
        shared_ptr[CBuffer] data
        shared_ptr[CBuffer] metadata
        shared_ptr[CRayObject] ray_object
        int64_t data_size

    with gil:
        for serialized_object in py_objects:
            data_size = serialized_object.total_bytes
            data = dynamic_pointer_cast[
                CBuffer, LocalMemoryBuffer](
                    make_shared[LocalMemoryBuffer](
                        <uint8_t*>NULL, data_size, True))
            stream = pyarrow.FixedSizeBufferWriter(
                pyarrow.py_buffer(Buffer.make(data)))
            serialized_object.write_to(stream)
            ray_object = make_shared[CRayObject](data, metadata)
            returns.push_back(ray_object)


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
            node_ip_address.encode("utf-8"), execute_task, True))

    def disconnect(self):
        with nogil:
            self.core_worker.get().Disconnect()

    def run_task_loop(self):
        with nogil:
            self.core_worker.get().Execution().Run()

    def get_current_task_id(self):
        return TaskID(self.core_worker.get().GetCurrentTaskId().Binary())

    def set_current_task_id(self, TaskID task_id):
        cdef:
            CTaskID c_task_id = task_id.native()

        with nogil:
            self.core_worker.get().SetCurrentTaskId(c_task_id)

    def get_current_job_id(self):
        return JobID(self.core_worker.get().GetCurrentJobId().Binary())

    def set_current_job_id(self, JobID job_id):
        cdef:
            CJobID c_job_id = job_id.native()

        with nogil:
            self.core_worker.get().SetCurrentJobId(c_job_id)

    def get_objects(self, object_ids, TaskID current_task_id):
        cdef:
            c_vector[shared_ptr[CRayObject]] results
            CTaskID c_task_id = current_task_id.native()
            c_vector[CObjectID] c_object_ids = ObjectIDsToVector(object_ids)

        with nogil:
            check_status(self.core_worker.get().Objects().Get(
                c_object_ids, -1, &results))

        return RayObjectsToDataMetadataPairs(results)

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

        # with profiling.profile("submit_task"):
        prepare_resources(resources, &c_resources)
        task_options = CTaskOptions(num_return_vals, c_resources)
        ray_function = CRayFunction(
            LANGUAGE_PYTHON, string_vector_from_list(function_descriptor))
        prepare_args(args, &args_vector)

        with nogil:
           check_status(self.core_worker.get().Tasks().SubmitTask(
               ray_function, args_vector, task_options, &return_ids))

        return VectorToObjectIDs(return_ids)

    def create_actor(self,
                     function_descriptor,
                     args,
                     uint64_t max_reconstructions,
                     resources,
                     placement_resources,
                     c_bool is_direct_call):
        cdef:
            ActorHandle actor_handle = ActorHandle.__new__(ActorHandle)
            CRayFunction ray_function
            c_vector[CTaskArg] args_vector
            c_vector[c_string] dynamic_worker_options
            unordered_map[c_string, double] c_resources
            unordered_map[c_string, double] c_placement_resources

        with profiling.profile("submit_task"):
            prepare_resources(resources, &c_resources)
            prepare_resources(placement_resources, &c_placement_resources)
            ray_function = CRayFunction(
                LANGUAGE_PYTHON, string_vector_from_list(function_descriptor))
            prepare_args(args, &args_vector)

            with nogil:
                check_status(self.core_worker.get().Tasks().CreateActor(
                    ray_function, args_vector,
                    CActorCreationOptions(
                        max_reconstructions, is_direct_call, c_resources,
                        c_placement_resources, dynamic_worker_options),
                    &actor_handle.inner))

            return actor_handle

    def submit_actor_task(self,
                          ActorHandle handle,
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

        with profiling.profile("submit_task"):
            prepare_resources(resources, &c_resources)
            task_options = CTaskOptions(num_return_vals, c_resources)
            ray_function = CRayFunction(
                LANGUAGE_PYTHON, string_vector_from_list(function_descriptor))
            prepare_args(args, &args_vector)

            with nogil:
                check_status(self.core_worker.get().Tasks().SubmitActorTask(
                      handle.inner.get()[0], ray_function,
                      args_vector, task_options, &return_ids))

            return VectorToObjectIDs(return_ids)

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
