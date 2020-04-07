package org.ray.runtime.task;

import com.google.common.base.Preconditions;
import java.util.List;
import java.util.stream.Collectors;
import org.ray.api.BaseActor;
import org.ray.api.id.ObjectId;
import org.ray.api.options.ActorCreationOptions;
import org.ray.api.options.CallOptions;
import org.ray.runtime.actor.NativeRayActor;
import org.ray.runtime.functionmanager.FunctionDescriptor;

/**
 * Task submitter for cluster mode. This is a wrapper class for core worker task interface.
 */
public class NativeTaskSubmitter implements TaskSubmitter {

  @Override
  public List<ObjectId> submitTask(FunctionDescriptor functionDescriptor, List<FunctionArg> args,
                                   int numReturns, CallOptions options) {
    List<byte[]> returnIds = nativeSubmitTask(functionDescriptor, args, numReturns, options);
    return returnIds.stream().map(ObjectId::new).collect(Collectors.toList());
  }

  @Override
  public BaseActor createActor(FunctionDescriptor functionDescriptor, List<FunctionArg> args,
                              ActorCreationOptions options) {
    byte[] actorId = nativeCreateActor(functionDescriptor, args, options);
    return NativeRayActor.create(actorId, functionDescriptor.getLanguage());
  }

  @Override
  public List<ObjectId> submitActorTask(
      BaseActor actor, FunctionDescriptor functionDescriptor,
      List<FunctionArg> args, int numReturns, CallOptions options) {
    Preconditions.checkState(actor instanceof NativeRayActor);
    List<byte[]> returnIds = nativeSubmitActorTask(actor.getId().getBytes(),
        functionDescriptor, args, numReturns, options);
    return returnIds.stream().map(ObjectId::new).collect(Collectors.toList());
  }

  private static native List<byte[]> nativeSubmitTask(FunctionDescriptor functionDescriptor,
      List<FunctionArg> args, int numReturns, CallOptions callOptions);

  private static native byte[] nativeCreateActor(FunctionDescriptor functionDescriptor,
      List<FunctionArg> args, ActorCreationOptions actorCreationOptions);

  private static native List<byte[]> nativeSubmitActorTask(byte[] actorId,
      FunctionDescriptor functionDescriptor, List<FunctionArg> args, int numReturns,
      CallOptions callOptions);
}
