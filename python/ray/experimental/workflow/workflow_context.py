import pathlib
from typing import Optional


class WorkflowStepContext:
    # TODO(suquark): we have to skip some type hints here, because we are
    # still not quite sure what should be the correct type. It will finalize
    # we we receive enough feedback about the API.
    def __init__(self,
                 workflow_id=None,
                 workflow_root_dir=None,
                 workflow_scope=None):
        """
        The structure for saving workflow step context. The context provides
        critical info (e.g. where to checkpoint, which is its parent step)
        for the step to execute correctly.

        Args:
            workflow_id: The workflow job ID.
            workflow_root_dir: The working directory of the workflow, used for
                checkpointing etc.
            workflow_scope: The "calling stack" of the current workflow step.
                It describe the parent workflow steps.
        """
        self.workflow_id = workflow_id
        self.workflow_root_dir = workflow_root_dir
        self.workflow_scope = workflow_scope or []

    def __reduce__(self):
        return WorkflowStepContext, (self.workflow_id, self.workflow_root_dir,
                                     self.workflow_scope)


_context: Optional[WorkflowStepContext] = None


def init_workflow_step_context(workflow_id, workflow_root_dir) -> None:
    global _context
    if workflow_root_dir is not None:
        workflow_root_dir = pathlib.Path(workflow_root_dir)
    else:
        workflow_root_dir = pathlib.Path.cwd() / ".rayflow"
    assert workflow_id is not None
    _context = WorkflowStepContext(workflow_id, workflow_root_dir)


def get_workflow_step_context() -> Optional[WorkflowStepContext]:
    return _context


def set_workflow_step_context(context: Optional[WorkflowStepContext]):
    global _context
    _context = context


def get_scope():
    return _context.workflow_scope


def get_workflow_root_dir():
    return _context.workflow_root_dir
