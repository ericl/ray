from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import threading
from collections import defaultdict

import boto3
from botocore.config import Config

from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
from ray.ray_constants import BOTO_MAX_RETRIES


import logging
logger = logging.getLogger(__name__)


def to_aws_format(tags):
    """Convert the Ray node name tag to the AWS-specific 'Name' tag."""

    if TAG_RAY_NODE_NAME in tags:
        tags["Name"] = tags[TAG_RAY_NODE_NAME]
        del tags[TAG_RAY_NODE_NAME]
    return tags


def from_aws_format(tags):
    """Convert the AWS-specific 'Name' tag to the Ray node name tag."""

    if "Name" in tags:
        tags[TAG_RAY_NODE_NAME] = tags["Name"]
        del tags["Name"]
    return tags


class AWSNodeProvider(NodeProvider):
    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        config = Config(retries={'max_attempts': BOTO_MAX_RETRIES})
        self.ec2 = boto3.resource(
            "ec2", region_name=provider_config["region"], config=config)

        # Try availability zones round-robin, starting from random offset
        self.subnet_idx = random.randint(0, 100)

        self.tag_cache = {}  # Tags that we believe to actually be on EC2.
        self.tag_cache_pending = {}  # Tags that we will soon upload.
        self.tag_cache_lock = threading.Lock()
        self.tag_cache_update_event = threading.Event()
        self.tag_cache_kill_event = threading.Event()
        self.tag_update_thread = threading.Thread(
            target=self._node_tag_update_loop
        )
        self.tag_update_thread.start()  # TODO: monitor this?

        # Cache of node objects from the last nodes() call. This avoids
        # excessive DescribeInstances requests.
        self.cached_nodes = {}

    def _node_tag_update_loop(self):
        while True:
            self.tag_cache_update_event.wait()
            self.tag_cache_update_event.clear()

            with self.tag_cache_lock:
                if self.tag_cache_pending:
                    DD = defaultdict(list)
                    for node_id, tags in self.tag_cache_pending.items():
                        for x in tags.items():
                            DD[x].append(node_id)

                    for (k, v), node_ids in DD.items():
                        logger.info(
                            "AWSNodeProvider: Setting tag {}={} on {}".format(
                                k, v, node_ids
                            )
                        )

                    for (k, v), node_ids in DD.items():
                        if k == TAG_RAY_NODE_NAME:  # TODO: to_aws_format
                            k = "Name"
                        self.ec2.meta.client.create_tags(
                            Resources=node_ids,
                            Tags=[{"Key": k, "Value": v}],
                        )

                    for node_id, tags in self.tag_cache_pending.items():
                        self.tag_cache[node_id].update(tags)
                    self.tag_cache_pending = {}

            self.tag_cache_kill_event.wait(timeout=5)
            if self.tag_cache_kill_event.is_set():
                return

    def nodes(self, tag_filters):
        # Note that these filters are acceptable because they are set on
        #       node initialization, and so can never be sitting in the cache.
        tag_filters = to_aws_format(tag_filters)
        filters = [
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running"],
            },
            {
                "Name": "tag:{}".format(TAG_RAY_CLUSTER_NAME),
                "Values": [self.cluster_name],
            },
        ]
        for k, v in tag_filters.items():
            filters.append({
                "Name": "tag:{}".format(k),
                "Values": [v],
            })

        nodes = list(self.ec2.instances.filter(Filters=filters))
        # Populate the tag cache with initial information if necessary
        for node in nodes:
            if node.id in self.tag_cache:
                continue

            self.tag_cache[node.id] = from_aws_format(
                { x["Key"]: x["Value"] for x in node.tags }
            )

        self.cached_nodes = { node.id: node for node in nodes }
        return [ node.id for node in nodes ]

    def is_running(self, node_id):
        node = self._node(node_id)
        return node.state["Name"] == "running"

    def is_terminated(self, node_id):
        node = self._node(node_id)
        state = node.state["Name"]
        return state not in ["running", "pending"]

    def node_tags(self, node_id):
        with self.tag_cache_lock:
            d1 = self.tag_cache[node_id]
            d2 = self.tag_cache_pending.get(node_id, {})
            return {**d1, **d2}

    def external_ip(self, node_id):
        return self._node(node_id).public_ip_address

    def internal_ip(self, node_id):
        return self._node(node_id).private_ip_address

    def set_node_tags(self, node_id, tags):
        logger.info("Setting tags {} for {}".format(tags, node_id))
        with self.tag_cache_lock:
            try:
                self.tag_cache_pending[node_id].update(tags)
            except KeyError:
                self.tag_cache_pending[node_id] = tags

            self.tag_cache_update_event.set()

    def create_node(self, node_config, tags, count):
        tags = to_aws_format(tags)
        conf = node_config.copy()
        tag_pairs = [{
            "Key": TAG_RAY_CLUSTER_NAME,
            "Value": self.cluster_name,
        }]
        for k, v in tags.items():
            tag_pairs.append({
                "Key": k,
                "Value": v,
            })
        tag_specs = [{
            "ResourceType": "instance",
            "Tags": tag_pairs,
        }]
        user_tag_specs = conf.get("TagSpecifications", [])
        # Allow users to add tags and override values of existing
        # tags with their own. This only applies to the resource type
        # "instance". All other resource types are appended to the list of
        # tag specs.
        for user_tag_spec in user_tag_specs:
            if user_tag_spec["ResourceType"] == "instance":
                for user_tag in user_tag_spec["Tags"]:
                    exists = False
                    for tag in tag_specs[0]["Tags"]:
                        if user_tag["Key"] == tag["Key"]:
                            exists = True
                            tag["Value"] = user_tag["Value"]
                            break
                    if not exists:
                        tag_specs[0]["Tags"] += [user_tag]
            else:
                tag_specs += [user_tag_spec]

        # SubnetIds is not a real config key: we must resolve to a
        # single SubnetId before invoking the AWS API.
        subnet_ids = conf.pop("SubnetIds")
        subnet_id = subnet_ids[self.subnet_idx % len(subnet_ids)]
        self.subnet_idx += 1
        conf.update({
            "MinCount": 1,
            "MaxCount": count,
            "SubnetId": subnet_id,
            "TagSpecifications": tag_specs
        })
        self.ec2.create_instances(**conf)

    def terminate_node(self, node_id):
        node = self._node(node_id)
        node.terminate()

        # self.cached_nodes.pop(node_id, None)  # TODO: Can we do this?
        self.tag_cache.pop(node_id, None)
        self.tag_cache_pending.pop(node_id, None)

    def terminate_nodes(self, node_ids):
        self.ec2.meta.client.terminate_instances(
            InstanceIds=node_ids
        )

        for node_id in node_ids:
            # self.cached_nodes.pop(node_id, None)  # TODO: Can we do this?
            self.tag_cache.pop(node_id, None)
            self.tag_cache_pending.pop(node_id, None)

    def _node(self, node_id):
        if node_id not in self.cached_nodes:
            self.nodes({})  # Side effect: should cache it.

        assert node_id in self.cached_nodes, "Invalid instance id {}".format(node_id)
        return self.cached_nodes[node_id]

    def cleanup(self):
        logger.info("Set kill event")
        self.tag_cache_update_event.set()
        self.tag_cache_kill_event.set()
