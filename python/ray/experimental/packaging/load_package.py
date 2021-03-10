"""Support for loading code packages into Ray at runtime.

Ray packages allow developers to define self-contained code modules that can
be imported reproducibly into any Ray cluster. Each package can define its own
runtime environment, which can include:
 - Different versions of code (e.g., from different git commits).
 - Different Python libraries (e.g., conda environments, pip dependencies).
 - Different Docker container images.

You can run this file for an example of loading a "hello world" package.
"""

import importlib.util
import os
import yaml

import ray
import ray._private.runtime_env as runtime_support


class _RuntimePackage:
    """Represents a loaded Ray package.

    This class provides access to the symbols defined by the stub file of the
    package (e.g., remote functions and actor definitions). You can also
    access the raw runtime env defined by the package via ``pkg._runtime_env``.
    """

    def __init__(self, name: str, desc: str, stub_file: str,
                 runtime_env: dict):
        self._name = name
        self._description = desc
        self._stub_file = stub_file
        self._runtime_env = runtime_env

        if not os.path.exists(stub_file):
            raise ValueError("Stub file does not exist: {}".format(stub_file))

        spec = importlib.util.spec_from_file_location(self._name,
                                                      self._stub_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

        for symbol in dir(self._module):
            if not symbol.startswith("_"):
                value = getattr(self._module, symbol)
                if (isinstance(value, ray.remote_function.RemoteFunction)
                        or isinstance(value, ray.actor.ActorClass)):
                    # TODO(ekl) use the runtime_env option here instead of
                    # the override env vars. Currently this doesn't work since
                    # there is no way to define per-task job config.
                    setattr(
                        self,
                        symbol,
                        value.options(override_environment_variables={
                            "RAY_RUNTIME_ENV_FILES": runtime_env["files"]
                        }))
                else:
                    setattr(self, symbol, value)

    def __repr__(self):
        return "ray._RuntimePackage(module={}, runtime_env={})".format(
            self._module, self._runtime_env)


def _download_from_github_if_needed(config_path: str) -> str:
    # TODO(ekl) support github URIs for the config.
    return config_path


def load_package(config_path: str) -> _RuntimePackage:
    """Load the code package given its config path.

    Args:
        config_path (str): The path to the configuration YAML that defines
            the package. For documentation on the packaging format, see the
            example YAML in ``example_pkg/ray_pkg.yaml``.

    Examples:
        >>> # Load from local.
        >>> my_pkg = load_package("~/path/to/my_pkg.yaml")

        >>> # Load from GitHub.
        >>> my_pkg = ray.util.load_package(
        ...   "https://github.com/demo/foo/blob/v3.0/project/my_pkg.yaml")

        >>> # Inspect the package runtime env.
        >>> print(my_pkg._runtime_env)
        ... {"conda": {...},
        ...  "docker": "anyscale-ml/ray-ml:nightly-py38-cpu",
        ...  "files": "https://github.com/demo/foo/blob/v3.0/project/"}

        >>> # Run remote functions from the package.
        >>> my_pkg.my_func.remote(1, 2)

        >>> # Create actors from the package.
        >>> actor = my_pkg.MyActor.remote(3, 4)

        >>> # Create new remote funcs in a package.
        >>> @ray.remote(runtime_env=my_pkg._runtime_env)
        >>> def f(): ...
    """

    if not os.path.exists(config_path):
        raise ValueError("Config file does not exist: {}".format(config_path))

    if not ray.is_initialized():
        # TODO(ekl) lift this requirement?
        raise RuntimeError("Ray must be initialized first to load packages.")

    config_path = _download_from_github_if_needed(config_path)
    # TODO(ekl) validate schema?
    config = yaml.safe_load(open(config_path).read())
    base_dir = os.path.abspath(os.path.dirname(config_path))
    runtime_env = config["runtime_env"]

    # Autofill working directory by uploading to GCS storage.
    if "files" not in runtime_env:
        pkg_name = runtime_support.get_project_package_name(
            working_dir=base_dir, modules=[])
        pkg_uri = runtime_support.Protocol.GCS.value + "://" + pkg_name
        if not runtime_support.package_exists(pkg_uri):
            tmp_path = "/tmp/ray/_tmp{}".format(pkg_name)
            runtime_support.create_project_package(
                working_dir=base_dir, modules=[], output_path=tmp_path)
            # TODO(ekl) does this get garbage collected correctly with the
            # current job id?
            runtime_support.push_package(pkg_uri, tmp_path)
            if not runtime_support.package_exists(pkg_uri):
                raise RuntimeError(
                    "Failed to upload package {}".format(pkg_uri))
            runtime_env["files"] = pkg_uri

    # Autofill conda config.
    conda_yaml = os.path.join(base_dir, "conda.yaml")
    if os.path.exists(conda_yaml):
        if "conda" in runtime_env:
            raise ValueError(
                "Both conda.yaml and conda: section found in package")
        runtime_env["conda"] = yaml.safe_load(open(conda_yaml).read())

    pkg = _RuntimePackage(
        name=config["name"],
        desc=config["description"],
        stub_file=os.path.join(base_dir, config["stub_file"]),
        runtime_env=runtime_env)
    return pkg


if __name__ == "__main__":
    ray.init()
    pkg = load_package("./example_pkg/ray_pkg.yaml")
    print("-> Loaded package", pkg)
    print("-> Package symbols", [x for x in dir(pkg) if not x.startswith("_")])
    print("-> Testing actor call")
    a = pkg.MyActor.remote()
    print(ray.get(a.f.remote()))
    print("-> Testing method call")
    for _ in range(5):
        print(ray.get(pkg.my_func.remote()))
