#ifndef RAY_CORE_WORKER_RAYLET_TRANSPORT_H
#define RAY_CORE_WORKER_RAYLET_TRANSPORT_H

#include <list>
#include <utility>

#include "ray/common/task/task.h"
#include "ray/core_worker/object_interface.h"
#include "ray/core_worker/transport/transport.h"
#include "ray/raylet/raylet_client.h"
#include "ray/rpc/worker/worker_server.h"

namespace ray {

class CoreWorkerRayletTaskReceiver : public CoreWorkerTaskReceiver,
                                     public rpc::WorkerTaskHandler {
 public:
  CoreWorkerRayletTaskReceiver(WorkerContext &worker_context,
                               std::unique_ptr<RayletClient> &raylet_client,
                               CoreWorkerObjectInterface &object_interface,
                               boost::asio::io_service &rpc_io_service,
                               boost::asio::io_service &main_io_service,
                               rpc::GrpcServer &server, const TaskHandler &task_handler);

  void HandleAssignTask(const rpc::AssignTaskRequest &request,
                        rpc::AssignTaskReply *reply,
                        rpc::SendReplyCallback send_reply_callback) override;

  void HandleStealTasks(const rpc::StealTasksRequest &request,
                        rpc::StealTasksReply *reply,
                        rpc::SendReplyCallback send_reply_callback) override;

 private:
  void ProcessAssignedTasks(rpc::SendReplyCallback send_reply_callback);

  Status HandleAssignTask0(const rpc::AssignTaskRequest &request,
                           const TaskSpecification &task_spec);

  // Worker context.
  WorkerContext &worker_context_;
  /// Raylet client.
  std::unique_ptr<RayletClient> &raylet_client_;
  // Object interface.
  CoreWorkerObjectInterface &object_interface_;
  /// The rpc service for `WorkerTaskService`.
  rpc::WorkerTaskGrpcService task_service_;
  /// The callback function to process a task.
  TaskHandler task_handler_;
  /// Event loop to run tasks on.
  boost::asio::io_service &task_main_io_service_;
  /// Mutex to protect the assigned task queues below.
  std::mutex mutex_;
  /// List of tasks to execute next.
  std::list<TaskSpecification> assigned_tasks_;
  /// Number of tasks assigned in total.
  int num_assigned_;
  /// RPC request being worked on.
  rpc::AssignTaskRequest assigned_req_;
};

}  // namespace ray

#endif  // RAY_CORE_WORKER_RAYLET_TRANSPORT_H
