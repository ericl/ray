from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import aiohttp.web
import argparse
import datetime
import json
import logging
import os
#import secrets
import socket
import threading
import traceback
import sys
import yaml

from pathlib import Path
from collections import Counter
from operator import itemgetter
from typing import Dict

import ray.ray_constants as ray_constants
import ray.utils

# Logger for this module. It should be configured at the entry point
# into the program using Ray. Ray provides a default configuration at
# entry/init points.
logger = logging.getLogger(__name__)


def to_unix_time(dt):
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


if sys.version_info[0] == 2:
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from SocketServer import TCPServer as HTTPServer
elif sys.version_info[0] == 3:
    from http.server import SimpleHTTPRequestHandler, HTTPServer


def make_handler(node_stats):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            routes = {
                "/res/main.css": self.handle_file,
                "/res/main.js": self.handle_file,
                "/": self.get_index,
                "/index.html": self.get_index,
                "/index.htm": self.get_index,
                "/api/node_info": self.node_info,
                "/api/super_client_table": self.node_info,
                "/api/ray_config": self.ray_config,
            }
            if self.path in routes:
                routes[self.path]()
            else:
                self.send_response(404)
        
        def handle_file(self, name=None):
            name = name or self.path
            path = os.path.dirname(os.path.abspath(__file__)) + name
            with open(path, "rb") as f:
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f.read())

        def get_index(self):
            return self.handle_file("/index.html")

        def node_info(self):
            now = datetime.datetime.utcnow()
            print("Querying stats")
            D = node_stats.get_node_stats()
            print("Stats", D)
            self.respond_json(D)

        def ray_config(self):
            try:
                with open(os.path.expanduser("~/ray_bootstrap_config.yaml")) as f:
                    cfg = yaml.load(f)
            except:
                return self.respond_json({"error": "No file found"})

            D = {
                "min_workers": cfg["min_workers"],
                "max_workers": cfg["max_workers"],
                "initial_workers": cfg["initial_workers"],
                "idle_timeout_minutes": cfg["idle_timeout_minutes"],
            }

            try:
                D["head_type"] = cfg["head_node"]["InstanceType"]
            except KeyError:
                D["head_type"] = "unknown"

            try:
                D["worker_type"] = cfg["worker_nodes"]["InstanceType"]
            except KeyError:
                D["worker_type"] = "unknown"

            self.respond_json(D)

        def respond_json(self, data):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "result": data,
                "error": None,
                "timestamp": to_unix_time(datetime.datetime.utcnow()),
            }).encode("utf-8"))

    return Handler


class NodeStats(threading.Thread):
    def __init__(self, redis_address, redis_password=None):
        self.redis_key = "{}.*".format(ray.gcs_utils.REPORTER_CHANNEL)
        self.redis_client = ray.services.create_redis_client(
            redis_address, password=redis_password)

        self._node_stats = {}
        self._node_stats_lock = threading.Lock()
        super().__init__()

    def calculate_totals(self) -> Dict:
        total_boot_time = 0
        total_cpus = 0
        total_workers = 0
        total_load = [0.0,0.0,0.0]
        total_storage_avail = 0
        total_storage_total = 0
        total_ram_avail = 0
        total_ram_total = 0
        total_sent = 0
        total_recv = 0

        for v in self._node_stats.values():
            total_boot_time += v["boot_time"]
            total_cpus += v["cpus"][0]
            total_workers += len(v["workers"])
            total_load[0] += v["load_avg"][0][0]
            total_load[1] += v["load_avg"][0][1]
            total_load[2] += v["load_avg"][0][2]
            total_storage_avail += v["disk"]["/"]["free"]
            total_storage_total += v["disk"]["/"]["total"]
            total_ram_avail += v["mem"][1]
            total_ram_total += v["mem"][0]
            total_sent += v["net"][0]
            total_recv += v["net"][1]

        return {
            "boot_time": total_boot_time,
            "n_workers": total_workers,
            "n_cores": total_cpus,
            "m_avail": total_ram_avail,
            "m_total": total_ram_total,
            "d_avail": total_storage_avail,
            "d_total": total_storage_total,
            "load": total_load,
            "n_sent": total_sent,
            "n_recv": total_recv,
        }

    def calculate_tasks(self) -> Counter:
        return Counter(
            (x["name"] for y in (v["workers"] for v in self._node_stats.values()) for x in y)
        )

    def purge_outdated_stats(self):
        def current(then, now):
            if (now - then) > 5:
                return False

            return True

        now = to_unix_time(datetime.datetime.utcnow())
        self._node_stats = {
            k: v
            for k, v in self._node_stats.items()
            if current(v["now"], now)
        }

    def get_node_stats(self) -> Dict:
        with self._node_stats_lock:
            self.purge_outdated_stats()
            node_stats = sorted(
                (v for v in self._node_stats.values()),
                key=itemgetter("boot_time")
            )
            return {
                "totals": self.calculate_totals(),
                "tasks": self.calculate_tasks(),
                "clients": node_stats,
            }

    def run(self):
        p = self.redis_client.pubsub()
        p.psubscribe(self.redis_key)
        logger.info("NodeStats: subscribed to {}".format(self.redis_key))

        for x in p.listen():
            if x["type"] != "pmessage":
                continue

            try:
                D = json.loads(x["data"])
                with self._node_stats_lock:
                    self._node_stats[D["hostname"]] = D
            except Exception:
                logger.exception(traceback.format_exc())
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Parse Redis server for the "
                     "dashboard to connect to."))
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="The HTTP port to serve from.")
    parser.add_argument(
        "--redis-address",
        required=True,
        type=str,
        help="The address to use for Redis.")
    parser.add_argument(
        "--redis-password",
        required=False,
        type=str,
        default=None,
        help="the password to use for Redis")
    parser.add_argument(
        "--logging-level",
        required=False,
        type=str,
        default=ray_constants.LOGGER_LEVEL,
        choices=ray_constants.LOGGER_LEVEL_CHOICES,
        help=ray_constants.LOGGER_LEVEL_HELP)
    parser.add_argument(
        "--logging-format",
        required=False,
        type=str,
        default=ray_constants.LOGGER_FORMAT,
        help=ray_constants.LOGGER_FORMAT_HELP)
    args = parser.parse_args()
    ray.utils.setup_logger(args.logging_level, args.logging_format)

    node_stats = NodeStats(args.redis_address, args.redis_password)
    node_stats.start()

    try:
        server = HTTPServer(("localhost", args.port), make_handler(node_stats))
        print("Started listening on", args.port)
        server.serve_forever()
    except Exception as e:
        # Something went wrong, so push an error to all drivers.
        redis_client = ray.services.create_redis_client(
            args.redis_address, password=args.redis_password)
        traceback_str = ray.utils.format_error_message(traceback.format_exc())
        message = ("The dashboard on node {} failed with the following "
                   "error:\n{}".format(os.uname()[1], traceback_str))
        ray.utils.push_error_to_driver_through_redis(
            redis_client, ray_constants.DASHBOARD_DIED_ERROR, message)
        raise e
