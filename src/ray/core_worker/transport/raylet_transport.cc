
#include "ray/core_worker/transport/raylet_transport.h"
#include "ray/common/common_protocol.h"
#include "ray/common/task/task.h"

namespace ray {

CoreWorkerRayletTaskReceiver::CoreWorkerRayletTaskReceiver(
    WorkerContext &worker_context, std::unique_ptr<RayletClient> &raylet_client,
    CoreWorkerObjectInterface &object_interface, boost::asio::io_service &rpc_io_service,
    boost::asio::io_service &main_io_service,
    rpc::GrpcServer &server, const TaskHandler &task_handler)
    : worker_context_(worker_context),
      raylet_client_(raylet_client),
      object_interface_(object_interface),
      task_service_(rpc_io_service, *this),
      task_handler_(task_handler),
      task_main_io_service_(main_io_service) {
  server.RegisterService(task_service_);
}

void CoreWorkerRayletTaskReceiver::HandleAssignTask(
    const rpc::AssignTaskRequest &request, rpc::AssignTaskReply *reply,
    rpc::SendReplyCallback send_reply_callback) {
  Status status;
  std::list<TaskSpecification> assigned;
  for (int i = 0; i < request.tasks_size(); i++) {
    Task task(request.tasks(i));
    const auto &task_spec = task.GetTaskSpecification();
    RAY_LOG(DEBUG) << "Received task " << task_spec.TaskId();
    if (task_spec.IsActorTask() && worker_context_.CurrentActorUseDirectCall()) {
      send_reply_callback(Status::Invalid("This actor only accepts direct calls."),
                          nullptr, nullptr);
      return;
    }
    assigned.push_back(task_spec);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    assigned_req_ = request;
    assigned_tasks_ = assigned;
    num_assigned_ = assigned.size();
  }

  // Let the main thread handle the actual task execution, so that we don't block
  // other RPCs such as StealTasks().
  task_main_io_service_.post(
      [this, send_reply_callback]() { ProcessAssignedTasks(send_reply_callback); });
}

void CoreWorkerRayletTaskReceiver::ProcessAssignedTasks(
    rpc::SendReplyCallback send_reply_callback) {
  // Process each assigned task in order.
  while (true) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (assigned_tasks_.empty()) {
      break;
    }
    auto task_spec = assigned_tasks_.front();
    assigned_tasks_.pop_front();
    // Don't hold the lock while processing the task. This allows other RPCs to
    // steal remaining tasks from us while we are processing this one.
    lock.unlock();
    HandleAssignTask0(assigned_req_, task_spec);
  }

  // Notify raylet that current task is done via a `TaskDone` message. This is to
  // ensure that the task is marked as finished by raylet only after previous
  // raylet client calls are completed. For example, if the worker sends a
  // NotifyUnblocked message that it is no longer blocked in a `ray.get`
  // on the normal raylet socket, then completes an assigned task, we
  // need to guarantee that raylet gets the former message first before
  // marking the task as completed. This is why a `TaskDone` message
  // is required - without it, it's possible that raylet receives
  // rpc reply first before the NotifyUnblocked message arrives,
  // as they use different connections, the `TaskDone` message is sent
  // to raylet via the same connection so the order is guaranteed.
  RAY_UNUSED(raylet_client_->TaskDone());
  // Send rpc reply.
  send_reply_callback(Status::OK(), nullptr, nullptr);
}

Status CoreWorkerRayletTaskReceiver::HandleAssignTask0(
    const rpc::AssignTaskRequest &request, const TaskSpecification &task_spec) {

  // Set the resource IDs for this task.
  // TODO: convert the resource map to protobuf and change this.
  ResourceMappingType resource_ids;
  auto resource_infos =
      flatbuffers::GetRoot<protocol::ResourceIdSetInfos>(request.resource_ids().data())
          ->resource_infos();
  for (size_t i = 0; i < resource_infos->size(); ++i) {
    auto const &fractional_resource_ids = resource_infos->Get(i);
    auto &acquired_resources =
        resource_ids[string_from_flatbuf(*fractional_resource_ids->resource_name())];

    size_t num_resource_ids = fractional_resource_ids->resource_ids()->size();
    size_t num_resource_fractions = fractional_resource_ids->resource_fractions()->size();
    RAY_CHECK(num_resource_ids == num_resource_fractions);
    RAY_CHECK(num_resource_ids > 0);
    for (size_t j = 0; j < num_resource_ids; ++j) {
      int64_t resource_id = fractional_resource_ids->resource_ids()->Get(j);
      double resource_fraction = fractional_resource_ids->resource_fractions()->Get(j);
      if (num_resource_ids > 1) {
        int64_t whole_fraction = resource_fraction;
        RAY_CHECK(whole_fraction == resource_fraction);
      }
      acquired_resources.push_back(std::make_pair(resource_id, resource_fraction));
    }
  }

  std::vector<std::shared_ptr<RayObject>> results;
  auto status = task_handler_(task_spec, resource_ids, &results);

  auto num_returns = task_spec.NumReturns();
  if (task_spec.IsActorCreationTask() || task_spec.IsActorTask()) {
    RAY_CHECK(num_returns > 0);
    // Decrease to account for the dummy object id.
    num_returns--;
  }

  RAY_LOG(DEBUG) << "Assigned task " << task_spec.TaskId()
                 << " finished execution. num_returns: " << num_returns;
  if (results.size() != 0) {
    RAY_CHECK(results.size() == num_returns);
    for (size_t i = 0; i < num_returns; i++) {
      ObjectID id = ObjectID::ForTaskReturn(
          task_spec.TaskId(), /*index=*/i + 1,
          /*transport_type=*/static_cast<int>(TaskTransportType::RAYLET));
      Status status = object_interface_.Put(*results[i], id);
      if (!status.ok()) {
        // NOTE(hchen): `PlasmaObjectExists` error is already ignored inside
        // `ObjectInterface::Put`, we treat other error types as fatal here.
        RAY_LOG(FATAL) << "Task " << task_spec.TaskId() << " failed to put object " << id
                       << " in store: " << status.message();
      } else {
        RAY_LOG(DEBUG) << "Task " << task_spec.TaskId() << " put object " << id
                       << " in store.";
      }
    }
  }

  return status;
}

void CoreWorkerRayletTaskReceiver::HandleStealTasks(
    const rpc::StealTasksRequest &request, rpc::StealTasksReply *reply,
    rpc::SendReplyCallback send_reply_callback) {
  std::unique_lock<std::mutex> lock(mutex_);
  int num_stolen = 0;
  // Avoid stealing all the tasks (steal up to N-1).
  while (!assigned_tasks_.empty() && num_stolen < num_assigned_ - 1) {
    reply->add_task_ids(assigned_tasks_.back().TaskId().Binary());
    assigned_tasks_.pop_back();
    num_stolen += 1;
  }
  send_reply_callback(Status::OK(), nullptr, nullptr);
}

}  // namespace ray
