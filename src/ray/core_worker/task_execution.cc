#include "ray/core_worker/task_execution.h"
#include "ray/core_worker/context.h"
#include "ray/core_worker/core_worker.h"
#include "ray/core_worker/transport/direct_actor_transport.h"
#include "ray/core_worker/transport/raylet_transport.h"

namespace ray {

CoreWorkerTaskExecutionInterface::CoreWorkerTaskExecutionInterface(
    WorkerContext &worker_context, std::unique_ptr<RayletClient> &raylet_client,
    CoreWorkerObjectInterface &object_interface,
    boost::asio::io_service &io_service,
    const std::shared_ptr<worker::Profiler> profiler,
    const NormalTaskCallback &normal_task_callback,
    const ActorTaskCallback &actor_task_callback)
    : worker_context_(worker_context),
      object_interface_(object_interface),
      profiler_(profiler),
      normal_task_callback_(normal_task_callback),
      actor_task_callback_(actor_task_callback),
      worker_server_("Worker", 0 /* let grpc choose port */),
      rpc_io_service_(io_service),
      main_service_(std::make_shared<boost::asio::io_service>()),
      main_work_(*main_service_) {
  RAY_CHECK(normal_task_callback_ != nullptr);
  RAY_CHECK(actor_task_callback_ != nullptr);

  auto func =
      std::bind(&CoreWorkerTaskExecutionInterface::ExecuteTask, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  task_receivers_.emplace(
      TaskTransportType::RAYLET,
      std::unique_ptr<CoreWorkerRayletTaskReceiver>(new CoreWorkerRayletTaskReceiver(
          worker_context_, raylet_client, object_interface_, rpc_io_service_,
          *main_service_, worker_server_, func)));
  task_receivers_.emplace(
      TaskTransportType::DIRECT_ACTOR,
      std::unique_ptr<CoreWorkerDirectActorTaskReceiver>(
          new CoreWorkerDirectActorTaskReceiver(worker_context_, object_interface_,
                                                rpc_io_service_, worker_server_, func)));

  // Start RPC server after all the task receivers are properly initialized.
  worker_server_.Run();
}

Status CoreWorkerTaskExecutionInterface::ExecuteTask(
    const TaskSpecification &task_spec, const ResourceMappingType &resource_ids,
    std::vector<std::shared_ptr<RayObject>> *results) {
  idle_profile_event_.reset();
  RAY_LOG(DEBUG) << "Executing task " << task_spec.TaskId();

  resource_ids_ = resource_ids;
  worker_context_.SetCurrentTask(task_spec);

  RayFunction func{task_spec.GetLanguage(), task_spec.FunctionDescriptor()};

  std::vector<TaskArg> args = BuildArgsForExecutor(task_spec);

  std::vector<ObjectID> return_ids;
  for (size_t i = 0; i < task_spec.NumReturns(); i++) {
    return_ids.push_back(task_spec.ReturnId(i));
  }

  Status status;
  if (task_spec.IsActorCreationTask() || task_spec.IsActorTask()) {
    RAY_CHECK(return_ids.size() > 0);
    return_ids.pop_back();
    ActorID actor_id = task_spec.IsActorCreationTask() ? task_spec.ActorCreationId()
                                                       : task_spec.ActorId();
    status = actor_task_callback_(
        func, worker_context_.GetCurrentJobID(), task_spec.TaskId(), actor_id,
        task_spec.IsActorCreationTask(),
        task_spec.GetRequiredResources().GetResourceMap(), args, return_ids, results);
  } else {
    status = normal_task_callback_(func, worker_context_.GetCurrentJobID(),
                                   task_spec.TaskId(), args, return_ids, results);
  }

  // TODO(zhijunfu):
  // 1. Check and handle failure.
  // 2. Save or load checkpoint.
  idle_profile_event_.reset(new worker::ProfileEvent(profiler_, "worker_idle"));
  return status;
}

void CoreWorkerTaskExecutionInterface::Run() {
  idle_profile_event_.reset(new worker::ProfileEvent(profiler_, "worker_idle"));
  main_service_->run();
}

void CoreWorkerTaskExecutionInterface::Stop() {
  // Stop main IO service.
  std::shared_ptr<boost::asio::io_service> main_service = main_service_;
  // Delay the execution of io_service::stop() to avoid deadlock if
  // CoreWorkerTaskExecutionInterface::Stop is called inside a task.
  main_service_->post([main_service]() { main_service->stop(); });
  idle_profile_event_.reset();
}

std::vector<TaskArg> CoreWorkerTaskExecutionInterface::BuildArgsForExecutor(
    const TaskSpecification &task) {
  std::vector<TaskArg> args;
  for (size_t i = 0; i < task.NumArgs(); ++i) {
    int count = task.ArgIdCount(i);
    if (count > 0) {
      // Passed by reference.
      RAY_CHECK(count == 1);
      args.push_back(TaskArg::PassByReference(task.ArgId(i, 0)));
    } else {
      // Passed by value.
      std::shared_ptr<LocalMemoryBuffer> data = nullptr;
      if (task.ArgDataSize(i)) {
        data = std::make_shared<LocalMemoryBuffer>(const_cast<uint8_t *>(task.ArgData(i)),
                                                   task.ArgDataSize(i));
      }
      std::shared_ptr<LocalMemoryBuffer> metadata = nullptr;
      if (task.ArgMetadataSize(i)) {
        metadata = std::make_shared<LocalMemoryBuffer>(
            const_cast<uint8_t *>(task.ArgMetadata(i)), task.ArgMetadataSize(i));
      }
      args.push_back(TaskArg::PassByValue(std::make_shared<RayObject>(data, metadata)));
    }
  }
  return args;
}

}  // namespace ray
