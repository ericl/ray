from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote
import logging
import os
import subprocess
import sys
import tempfile
import time

from threading import Thread

from ray.autoscaler.node_provider import get_node_provider
from ray.autoscaler.tags import TAG_RAY_NODE_STATUS, TAG_RAY_RUNTIME_CONFIG

# How long to wait for a node to start, in seconds
NODE_START_WAIT_S = 300
SSH_CHECK_INTERVAL = 5
SSH_CONTROL_PATH = "/tmp/ray_ssh_sockets"

def get_default_ssh_options(private_key, connect_timeout):
    OPTS = [
        ("ConnectTimeout", "{}s".format(connect_timeout)),
        ("StrictHostKeyChecking", "no"),
        ("ControlMaster", "auto"),
        ("ControlPath", "{}/%C".format(SSH_CONTROL_PATH)),
        ("ControlPersist", "yes"),
    ]

    return ["-i", private_key] + [
        x
        for y in ( ["-o", "{}={}".format(k, v)] for k, v in OPTS )
        for x in y
    ]

logger = logging.getLogger(__name__)


def pretty_cmd(cmd_str):
    return "\n\n\t{}\n\n".format(cmd_str)


class NodeUpdater(object):
    """A process for syncing files and running init commands on a node."""

    def __init__(self,
                 node_id,
                 provider_config,
                 provider,
                 auth_config,
                 cluster_name,
                 file_mounts,
                 setup_cmds,
                 runtime_hash,
                 redirect_output=True,
                 process_runner=subprocess,
                 use_internal_ip=False):
        self.daemon = True
        self.process_runner = process_runner
        self.node_id = node_id
        self.use_internal_ip = (use_internal_ip or provider_config.get(
            "use_internal_ips", False))
        self.provider = provider
        self.ssh_private_key = auth_config["ssh_private_key"]
        self.ssh_user = auth_config["ssh_user"]
        self.ssh_ip = None
        self.file_mounts = {
            remote: os.path.expanduser(local)
            for remote, local in file_mounts.items()
        }
        self.setup_cmds = setup_cmds
        self.runtime_hash = runtime_hash
        self.logger = logger.getChild(str(node_id))
        if redirect_output:
            self.logfile = tempfile.NamedTemporaryFile(
                mode="w", prefix="node-updater-", delete=False)
            handler = logging.StreamHandler(stream=self.logfile)
            handler.setLevel(logging.INFO)
            self.logger.addHandler(handler)
            self.output_name = self.logfile.name
            self.stdout = self.logfile
            self.stderr = self.logfile
        else:
            self.logfile = None
            self.output_name = "(console)"
            self.stdout = sys.stdout
            self.stderr = sys.stderr

    def get_caller(self, check_error):
        if check_error:
            return self.process_runner.call
        else:
            return self.process_runner.check_call

    def get_node_ip(self):
        if self.use_internal_ip:
            return self.provider.internal_ip(self.node_id)
        else:
            return self.provider.external_ip(self.node_id)

    def set_ssh_ip_if_required(self):
        if self.ssh_ip is not None:
            return

        # We assume that this never changes.
        #   I think that's reasonable.
        deadline = time.time() + NODE_START_WAIT_S
        while time.time() < deadline and \
                not self.provider.is_terminated(self.node_id):
            self.logger.info("NodeUpdater: Waiting for IP of {}...".format(
                self.node_id))
            self.ssh_ip = self.get_node_ip()
            if self.ssh_ip is not None:
                break
            time.sleep(10)

        assert self.ssh_ip is not None, "Unable to find IP of node"

        # This should run before any SSH commands and therefore ensure that
        #   the ControlPath directory exists, allowing SSH to maintain
        #   persistent sessions later on.
        with open("/dev/null", "w") as redirect:
            self.get_caller(False)(
                ["mkdir", "-p", SSH_CONTROL_PATH],
                stdout=redirect,
                stderr=redirect)

            self.get_caller(False)(
                ["chmod", "0700", SSH_CONTROL_PATH],
                stdout=redirect,
                stderr=redirect)

    def run(self):
        self.logger.info(
            "NodeUpdater: Updating {} to {}, logging to {}".format(
                self.node_id, self.runtime_hash, self.output_name))
        try:
            self.do_update()
        except Exception as e:
            error_str = str(e)
            if hasattr(e, "cmd"):
                error_str = "(Exit Status {}) {}".format(
                    e.returncode, pretty_cmd(" ".join(e.cmd)))
            self.logger.error("NodeUpdater: Error updating {}"
                              "See {} for remote logs.".format(
                                  error_str, self.output_name))
            self.provider.set_node_tags(self.node_id,
                                        {TAG_RAY_NODE_STATUS: "update-failed"})
            if self.logfile is not None:
                self.logger.info("----- BEGIN REMOTE LOGS -----\n" +
                                 open(self.logfile.name).read() +
                                 "\n----- END REMOTE LOGS -----")
            raise e
        self.provider.set_node_tags(
            self.node_id, {
                TAG_RAY_NODE_STATUS: "up-to-date",
                TAG_RAY_RUNTIME_CONFIG: self.runtime_hash
            })
        self.logger.info("NodeUpdater: Applied config {} to node {}".format(
            self.runtime_hash, self.node_id))

        self.exitcode = 0

    def do_update(self):
        self.provider.set_node_tags(self.node_id,
                                    {TAG_RAY_NODE_STATUS: "waiting-for-ssh"})

        deadline = time.time() + NODE_START_WAIT_S
        self.set_ssh_ip_if_required()

        # Wait for SSH access
        ssh_ok = False
        while time.time() < deadline and \
                not self.provider.is_terminated(self.node_id):
            try:
                self.logger.info(
                    "NodeUpdater: Waiting for SSH to {}...".format(
                        self.node_id))
                if not self.provider.is_running(self.node_id):
                    raise Exception("Node not running yet...")
                self.ssh_cmd(
                    "uptime",
                    connect_timeout=5,
                    redirect=open("/dev/null", "w"))
                ssh_ok = True
            except Exception as e:
                retry_str = str(e)
                if hasattr(e, "cmd"):
                    retry_str = "(Exit Status {}): {}".format(
                        e.returncode, pretty_cmd(" ".join(e.cmd)))
                self.logger.debug(
                    "NodeUpdater: SSH not up, retrying: {}".format(retry_str),
                )
                time.sleep(SSH_CHECK_INTERVAL)
            else:
                break
        assert ssh_ok, "Unable to SSH to node"

        # Rsync file mounts
        self.provider.set_node_tags(self.node_id,
                                    {TAG_RAY_NODE_STATUS: "syncing-files"})
        for remote_path, local_path in self.file_mounts.items():
            self.logger.info("NodeUpdater: Syncing {} to {}...".format(
                local_path, remote_path))
            assert os.path.exists(local_path), local_path
            if os.path.isdir(local_path):
                if not local_path.endswith("/"):
                    local_path += "/"
                if not remote_path.endswith("/"):
                    remote_path += "/"
            self.ssh_cmd("mkdir -p {}".format(os.path.dirname(remote_path)))
            self.rsync_up(local_path, remote_path)

        # Run init commands
        self.provider.set_node_tags(self.node_id,
                                    {TAG_RAY_NODE_STATUS: "setting-up"})
        for cmd in self.setup_cmds:
            self.ssh_cmd(cmd, verbose=True)

    def rsync_up(self, source, target, check_error=True):
        self.set_ssh_ip_if_required()
        self.get_caller(check_error)(
            [
                "rsync", "-e", " ".join(["ssh"] +
                    get_default_ssh_options(self.ssh_private_key, 120)),
                "--delete", "-avz", source,
                "{}@{}:{}".format(self.ssh_user, self.ssh_ip, target)
            ],
            stdout=self.stdout,
            stderr=self.stderr)

    def rsync_down(self, source, target, check_error=True):
        self.set_ssh_ip_if_required()
        self.get_caller(check_error)(
            [
                "rsync", "-e", " ".join(["ssh"] +
                    get_default_ssh_options(self.ssh_private_key, 120)),
                "-avz",
                "{}@{}:{}".format(self.ssh_user, self.ssh_ip, source), target
            ],
            stdout=self.stdout,
            stderr=self.stderr)

    def ssh_cmd(self,
                cmd,
                connect_timeout=120,
                redirect=None,
                verbose=False,
                allocate_tty=False,
                emulate_interactive=True,
                expect_error=False,
                port_forward=None):

        self.set_ssh_ip_if_required()

        if verbose:
            self.logger.info("NodeUpdater: running {} on {}...".format(
                pretty_cmd(cmd), self.ssh_ip))
        ssh = ["ssh"]
        if allocate_tty:
            ssh.append("-tt")
        if emulate_interactive:
            force_interactive = (
                "set -i || true && source ~/.bashrc && "
                "export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && ")
            cmd = "bash --login -c {}".format(quote(force_interactive + cmd))

        if port_forward is None:
            ssh_opt = []
        else:
            ssh_opt = [
                "-L", "{}:localhost:{}".format(port_forward, port_forward)
            ]

        self.get_caller(expect_error)(
            ssh + ssh_opt +
            get_default_ssh_options(self.ssh_private_key, connect_timeout) + [
                "{}@{}".format(self.ssh_user, self.ssh_ip), cmd
            ],
            stdout=redirect or self.stdout,
            stderr=redirect or self.stderr)

class NodeUpdaterThread(NodeUpdater, Thread):
    def __init__(self, *args, **kwargs):
        Thread.__init__(self)
        NodeUpdater.__init__(self, *args, **kwargs)
        self.exitcode = -1
