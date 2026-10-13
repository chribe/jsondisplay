import argparse
import colorsys
import copy
import fnmatch
import hashlib
import json
import time
import webbrowser
from threading import Timer

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update


# =========================================================
# Configuration
# =========================================================
DEFAULT_EXCLUDE_PATTERNS = [
    "*transform*",
    "NXLog",
]

MAX_EDITABLE_LIST_ITEMS = 50
MAX_TREE_ITEMS = 100
MAX_HISTORY_LENGTH = 30
DOUBLE_CLICK_SECONDS = 0.40

# Keeps the 3D camera view during Dash updates.
PLOT_UIREVISION = "nexus-json-geometry-viewer-v5"
PLOT_SCENE_UIREVISION = "nexus-3d-camera-preserve-v1"


# =========================================================
# General utilities
# =========================================================
def attrs_to_dict(attrs):
    result = {}

    if not isinstance(attrs, list):
        return result

    for attribute in attrs:
        if not isinstance(attribute, dict):
            continue

        name = attribute.get("name")

        if name is not None:
            result[name] = attribute.get("values")

    return result


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)

    if length == 0:
        return vector

    return vector / length


def component_color(nx_class):
    """Generate a stable colour for every NX_class."""
    nx_class = str(nx_class or "unknown")

    digest = hashlib.md5(
        nx_class.encode("utf-8")
    ).hexdigest()

    hue = int(digest[:8], 16) / 0xFFFFFFFF

    red, green, blue = colorsys.hsv_to_rgb(
        hue,
        0.70,
        0.85,
    )

    return (
        f"rgb({int(red * 255)}, "
        f"{int(green * 255)}, "
        f"{int(blue * 255)})"
    )


def parse_exclude_patterns(text):
    """Parse comma-, semicolon-, or newline-separated patterns."""
    if not text:
        return []

    text = text.replace(";", ",").replace("\n", ",")

    return [
        pattern.strip()
        for pattern in text.split(",")
        if pattern.strip()
    ]


def component_matches_patterns(component, patterns):
    """Match wildcard patterns against name, NX_class and path."""
    if not patterns:
        return False

    candidates = [
        str(component["name"]),
        str(component["nx_class"]),
        str(component["path"]),
    ]

    for pattern in patterns:
        pattern = str(pattern).lower()

        for candidate in candidates:
            if fnmatch.fnmatchcase(
                candidate.lower(),
                pattern,
            ):
                return True

    return False


def validate_limits(limits):
    """Validate that a minimum limit does not exceed a maximum."""
    for axis in ("x", "y", "z"):
        minimum = limits[f"{axis}min"]
        maximum = limits[f"{axis}max"]

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"{axis}min ({minimum}) must not exceed "
                f"{axis}max ({maximum})."
            )

    return limits


def point_in_limits(x, y, z, limits):
    if limits["xmin"] is not None and x < limits["xmin"]:
        return False
    if limits["xmax"] is not None and x > limits["xmax"]:
        return False
    if limits["ymin"] is not None and y < limits["ymin"]:
        return False
    if limits["ymax"] is not None and y > limits["ymax"]:
        return False
    if limits["zmin"] is not None and z < limits["zmin"]:
        return False
    if limits["zmax"] is not None and z > limits["zmax"]:
        return False

    return True


def point_mask(points, limits):
    mask = np.ones(len(points), dtype=bool)

    if limits["xmin"] is not None:
        mask &= points[:, 0] >= limits["xmin"]

    if limits["xmax"] is not None:
        mask &= points[:, 0] <= limits["xmax"]

    if limits["ymin"] is not None:
        mask &= points[:, 1] >= limits["ymin"]

    if limits["ymax"] is not None:
        mask &= points[:, 1] <= limits["ymax"]

    if limits["zmin"] is not None:
        mask &= points[:, 2] >= limits["zmin"]

    if limits["zmax"] is not None:
        mask &= points[:, 2] <= limits["zmax"]

    return mask


def downsample_points(point_set, max_points):
    """Limit point clouds before sending them to the browser."""
    count = len(point_set["x"])

    if max_points is None or count <= max_points:
        return point_set

    indices = np.linspace(
        0,
        count - 1,
        max_points,
        dtype=int,
    )

    print(
        f"Downsampling {point_set['name']}: "
        f"{count} -> {max_points} points"
    )

    return {
        "name": point_set["name"],
        "x": point_set["x"][indices],
        "y": point_set["y"][indices],
        "z": point_set["z"][indices],
    }


def path_key(json_path):
    """Return a serialisable key for a JSON path."""
    return json.dumps(json_path)


def parent_path_keys(json_path):
    """Return the root and all parent JSON paths."""
    return [
        path_key(json_path[:index])
        for index in range(len(json_path) + 1)
    ]


def normalise_nexus_path(path):
    """Canonicalise paths so references and indexed paths use the same root.

    Some JSON exports wrap the NeXus tree in an extra root node (for example,
    ``/root/entry/instrument/...``), while ``depends_on`` attributes use the
    canonical NeXus path (``/entry/instrument/...``).  When an ``entry`` path
    segment exists, discard any wrapper segments before it.
    """
    if path in {None, "", "."}:
        return None

    parts = [part for part in str(path).strip().split("/") if part]
    if "entry" in parts:
        parts = parts[parts.index("entry"):]

    return "/" + "/".join(parts)


def get_value_at_path(data, json_path):
    current = data

    for key in json_path:
        current = current[key]

    return current


def set_value_at_path(data, json_path, value):
    current = data

    for key in json_path[:-1]:
        current = current[key]

    current[json_path[-1]] = value


def parse_edited_value(text, old_value):
    """
    Convert editable text to JSON data.

    Numeric values may switch freely, for example:
        1.5 -> 10 -> 1.5
    """
    text = str(text).strip()

    if isinstance(old_value, bool):
        if text.lower() in {"true", "1", "yes", "on"}:
            return True

        if text.lower() in {"false", "0", "no", "off"}:
            return False

        raise ValueError("Expected boolean input.")

    if old_value is None:
        if text.lower() == "null":
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    if (
        isinstance(old_value, (int, float, np.integer, np.floating))
        and not isinstance(old_value, bool)
    ):
        number = float(text)

        if "." not in text and "e" not in text.lower():
            return int(number)

        return number

    if isinstance(old_value, (list, dict)):
        return json.loads(text)

    return text


# =========================================================
# Transformation matrices
# =========================================================
def rotation_matrix(axis, angle_deg):
    axis = normalize(axis)
    angle = np.deg2rad(angle_deg)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def translation_matrix(vector, value):
    matrix = np.eye(4)
    matrix[:3, 3] = value * np.asarray(vector, dtype=float)
    return matrix


def homogeneous_rotation(axis, angle_deg):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix(axis, angle_deg)
    return matrix


# =========================================================
# NeXus components and transformations
# =========================================================
def collect_components(
    node,
    path="",
    json_path=None,
    components=None,
):
    """
    Collect all groups carrying an NX_class attribute.

    `path` is the generated NeXus-style absolute path.
    `json_path` identifies the raw location in the JSON tree.
    """
    if components is None:
        components = []

    if json_path is None:
        json_path = []

    if not isinstance(node, dict):
        return components

    name = node.get("name", "unnamed")

    nx_class = attrs_to_dict(
        node.get("attributes", [])
    ).get("NX_class")

    current_path = f"{path}/{name}"

    if nx_class is not None:
        components.append({
            "name": name,
            "node": node,
            "path": current_path,
            "json_path": json_path,
            "nx_class": str(nx_class),
            "trace_indices": [],
        })

    for index, child in enumerate(node.get("children", [])):
        collect_components(
            child,
            current_path,
            json_path + ["children", index],
            components,
        )

    return components



def transform_to_matrix(transform):
    vector = transform.get("vector")
    value = transform.get("value")

    if vector is None or value is None:
        return np.eye(4)

    try:
        vector = np.asarray(vector, dtype=float)
        value = float(value)
    except (TypeError, ValueError):
        return np.eye(4)

    if transform.get("type") == "translation":
        return translation_matrix(vector, value)

    if transform.get("type") == "rotation":
        if transform.get("units") in {"radians", "rad"}:
            value = np.rad2deg(value)

        return homogeneous_rotation(vector, value)

    return np.eye(4)


def collect_transformations(component):
    """Return a component's transformations keyed by dataset name.

    Each record uses the same fields as the global transform resolver.
    """
    result = {}
    node = component.get("node", {})

    for child in node.get("children", []):
        if child.get("type") != "group" or child.get("name") != "transformations":
            continue

        for item in child.get("children", []):
            if item.get("module") != "dataset":
                continue

            config = item.get("config", {})
            name = config.get("name")
            if not name:
                continue

            attrs = attrs_to_dict(item.get("attributes", []))
            result[name] = {
                "name": name,
                "value": config.get("values"),
                "type": attrs.get("transformation_type"),
                "vector": attrs.get("vector"),
                "units": attrs.get("units"),
                "depends_on": attrs.get("depends_on"),
            }

    return result


def component_terminal_transform(component, transforms=None):
    """Find the terminal transformation driving a component."""
    if transforms is None:
        transforms = collect_transformations(component)

    if not transforms:
        return None

    depends_on = None

    for child in component["node"].get("children", []):
        if child.get("module") != "dataset":
            continue

        config = child.get("config", {})

        if config.get("name") == "depends_on":
            depends_on = config.get("values")
            break

    if depends_on not in {None, "", "."}:
        return str(depends_on).split("/")[-1]

    referenced = {
        str(transform["depends_on"]).split("/")[-1]
        for transform in transforms.values()
        if transform.get("depends_on")
    }

    terminal = [
        name
        for name in transforms
        if name not in referenced
    ]

    return terminal[0] if terminal else None

# =========================================================
# Global transformation index
# =========================================================
def transform_absolute_path(component_path, transform_name):
    """
    Construct a full absolute NeXus path for a transformation.

    Example:
        /entry/instrument/An0/transformations/x_translation
    """
    return (
        f"{component_path}/transformations/{transform_name}"
    )


def build_global_transform_index(components):
    """
    Build a global lookup table of all transformations in all
    components.

    Keys are full NeXus paths.
    """
    index = {}

    for component in components:
        node = component["node"]

        for child in node.get("children", []):
            if (
                child.get("type") != "group"
                or child.get("name") != "transformations"
            ):
                continue

            for transform in child.get("children", []):
                if transform.get("module") != "dataset":
                    continue

                config = transform.get("config", {})
                transform_name = config.get("name")

                if transform_name is None:
                    continue

                attrs = attrs_to_dict(
                    transform.get("attributes", [])
                )

                # Pass the full component record: the active helper
                # expects a component dictionary and normalises its path.
                absolute_path = transform_absolute_path(
                    component,
                    transform_name,
                )

                index[absolute_path] = {
                    "path": absolute_path,
                    "component_path": component["path"],
                    "component_name": component["name"],
                    "name": transform_name,
                    "value": config.get("values"),
                    "type": attrs.get(
                        "transformation_type"
                    ),
                    "vector": attrs.get("vector"),
                    "units": attrs.get("units"),
                    "depends_on": attrs.get("depends_on"),
                }

    return index

# =========================================================
# Transformation -> matrix
# =========================================================
def transform_to_matrix(transform):
    transform_type = transform.get("type")
    vector = transform.get("vector")
    value = transform.get("value")
    units = transform.get("units")

    if vector is None or value is None:
        return np.eye(4)

    try:
        vector = np.asarray(vector, dtype=float)
        value = float(value)
    except (TypeError, ValueError):
        return np.eye(4)

    if transform_type == "translation":
        return translation_matrix(vector, value)

    if transform_type == "rotation":
        if units in {"radians", "rad"}:
            value = np.rad2deg(value)

        return homogeneous_rotation(vector, value)

    return np.eye(4)

# =========================================================
# Resolve depends_on references globally
# =========================================================
def resolve_depends_on_path(
    depends_on,
    current_component_path,
):
    """
    Resolve a depends_on reference into a full NeXus path.

    Supports:
      - "."
      - absolute NeXus paths
      - local transform names
      - relative paths
    """
    if depends_on in {None, "", "."}:
        return None

    depends_on = str(depends_on).strip()

    # Already absolute: canonicalise it to the same root used by the index.
    if depends_on.startswith("/"):
        return normalise_nexus_path(depends_on)

    # Local transform name
    if "/" not in depends_on:
        return (
            f"{current_component_path}/transformations/"
            f"{depends_on}"
        )

    # Relative path
    return (
        f"{current_component_path}/"
        f"{depends_on.lstrip('/')}"
    )
# =========================================================
# Global recursive chain resolution
# =========================================================
def resolve_transform_chain_global(
    transform_index,
    current_transform_path,
    visited=None,
):
    """
    Resolve a transformation chain globally across components.

    Unlike the old implementation, this follows full NeXus
    depends_on paths and therefore supports inter-component
    dependencies correctly.
    """
    if visited is None:
        visited = set()

    if current_transform_path in {None, "", "."}:
        return np.eye(4)

    if current_transform_path in visited:
        print(
            "WARNING: transformation dependency loop at "
            f"'{current_transform_path}'"
        )
        return np.eye(4)

    visited.add(current_transform_path)

    transform = transform_index.get(
        current_transform_path
    )

    if transform is None:
        print(
            "WARNING: transform not found:\n"
            f"  {current_transform_path}"
        )
        return np.eye(4)

    dependency_path = resolve_depends_on_path(
        transform.get("depends_on"),
        transform["component_path"],
    )

    parent_matrix = resolve_transform_chain_global(
        transform_index,
        dependency_path,
        visited,
    )

    current_matrix = transform_to_matrix(transform)

    return parent_matrix @ current_matrix


# =========================================================
# Component matrix using global dependencies
# =========================================================
def build_component_matrix(
    component,
    transform_index,
):
    """
    Build the global transformation matrix for one component.

    This version correctly follows depends_on chains across
    components.
    """
    node = component["node"]

    component_depends_on = None

    for child in node.get("children", []):
        if child.get("module") != "dataset":
            continue

        config = child.get("config", {})

        if config.get("name") == "depends_on":
            component_depends_on = config.get("values")
            break

    if component_depends_on in {None, "", "."}:
        return np.eye(4)

    terminal_path = resolve_depends_on_path(
        component_depends_on,
        component["path"],
    )

    return resolve_transform_chain_global(
        transform_index,
        terminal_path,
    )


# =========================================================
# Geometry extraction
# =========================================================
def extract_child_points(component, matrix, limits=None):
    datasets = {}
    point_sets = []

    for child in component["node"].get("children", []):
        if child.get("module") != "dataset":
            continue

        config = child.get("config", {})
        name = config.get("name")

        if name is not None:
            datasets[name] = config.get("values")

    candidates = [
        ("x_pixel_offset", "y_pixel_offset", "z_pixel_offset"),
        ("x_offset", "y_offset", "z_offset"),
    ]

    for x_name, y_name, z_name in candidates:
        if not all(
            name in datasets
            for name in (x_name, y_name, z_name)
        ):
            continue

        try:
            x = np.asarray(datasets[x_name], dtype=float).flatten()
            y = np.asarray(datasets[y_name], dtype=float).flatten()
            z = np.asarray(datasets[z_name], dtype=float).flatten()
        except (TypeError, ValueError):
            continue

        if not (len(x) == len(y) == len(z)):
            print(
                f"WARNING: unequal offset lengths for "
                f"'{component['name']}'."
            )
            continue

        local_points = np.column_stack((x, y, z))
        homogeneous_points = np.column_stack((
            local_points,
            np.ones(len(local_points)),
        ))

        points = (matrix @ homogeneous_points.T).T[:, :3]

        if limits is not None:
            points = points[point_mask(points, limits)]

        point_sets.append({
            "name": f"{component['name']} children",
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
        })

    return point_sets


# =========================================================
# Plot generation
# =========================================================
def plot_instrument(
    data,
    json_file,
    exclude_patterns=None,
    visible_paths=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    zmin=None,
    zmax=None,
    max_points=100_000,
):
    if exclude_patterns is None:
        exclude_patterns = []

    limits = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }

    components = collect_components(data)
    transform_index = build_global_transform_index(
    components
    )
    figure = go.Figure()

    x_abs_max = 0.0
    y_abs_max = 0.0

    for component in components:
        component["trace_indices"] = []

        excluded = component_matches_patterns(
            component,
            exclude_patterns,
        )

        if visible_paths is None:
            visible = not excluded
        else:
            visible = (
                component["path"] in visible_paths
                and not excluded
            )

        matrix = build_component_matrix(
            component,
            transform_index,
        )
        origin = matrix @ np.array([0, 0, 0, 1], dtype=float)

        x0, y0, z0 = origin[:3]

        if not point_in_limits(x0, y0, z0, limits):
            continue

        x_abs_max = max(x_abs_max, abs(float(x0)))
        y_abs_max = max(y_abs_max, abs(float(y0)))

        colour = component_color(component["nx_class"])

        component["trace_indices"].append(len(figure.data))

        figure.add_trace(
            go.Scatter3d(
                x=[x0],
                y=[y0],
                z=[z0],
                mode="markers+text",
                name=component["name"],
                text=[component["name"]],
                textposition="top center",
                visible=visible,
                meta={
                    "component_path": component["path"],
                },
                marker=dict(
                    size=8,
                    color=colour,
                    line=dict(color="black", width=1),
                ),
                hovertext=[
                    (
                        f"<b>{component['name']}</b><br>"
                        f"NX class: {component['nx_class']}<br>"
                        f"Path: {component['path']}<br>"
                        f"Position: ({x0:.2f}, {y0:.2f}, {z0:.2f})"
                    )
                ],
                hoverinfo="text",
            )
        )

        for point_set in extract_child_points(
            component,
            matrix,
            limits,
        ):
            if len(point_set["x"]) == 0:
                continue

            point_set = downsample_points(
                point_set,
                max_points,
            )

            x_abs_max = max(
                x_abs_max,
                float(np.max(np.abs(point_set["x"]))),
            )

            y_abs_max = max(
                y_abs_max,
                float(np.max(np.abs(point_set["y"]))),
            )

            component["trace_indices"].append(len(figure.data))

            figure.add_trace(
                go.Scatter3d(
                    x=point_set["x"],
                    y=point_set["y"],
                    z=point_set["z"],
                    mode="markers",
                    name=point_set["name"],
                    visible=visible,
                    meta={
                        "component_path": component["path"],
                    },
                    marker=dict(
                        size=1.5,
                        color=colour,
                        opacity=0.25,
                    ),
                    hoverinfo="none",
                )
            )

    x_abs_max = max(x_abs_max, 1.0)
    y_abs_max = max(y_abs_max, 1.0)

    figure.update_layout(
        title=f"NeXus JSON Geometry: {json_file}",
        uirevision=PLOT_UIREVISION,
        scene=dict(
            xaxis=dict(
                title="X [m]",
                range=[-x_abs_max, x_abs_max],
            ),
            yaxis=dict(
                title="Y [m]",
                range=[-y_abs_max, y_abs_max],
            ),
            zaxis=dict(
                title="Z [m]",
            ),

            # Fixed equal visual X:Y:Z ratio.
            aspectmode="cube",

            # Keep user-controlled 3D camera changes across figure updates.
            uirevision=PLOT_SCENE_UIREVISION,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )

    return figure, components


# =========================================================
# Correct full-path depends_on dependency resolution
# =========================================================
def transform_absolute_path(component, transform_name):
    """
    Generated absolute transformation path.

    Example:
        /entry/instrument/detector/transformations/location
    """
    return normalise_nexus_path(
        f"{component['path']}/transformations/{transform_name}"
    )


def build_transform_index(components):
    """
    Index every transform by its generated full absolute NeXus path.

    The index retains the owning component and original depends_on
    value so dependencies can be resolved without comparing only
    transform basenames.
    """
    index = {}

    for component in components:
        transforms = collect_transformations(component)

        for name, transform in transforms.items():
            absolute_path = transform_absolute_path(
                component,
                name,
            )

            index[absolute_path] = {
                "path": absolute_path,
                "owner_path": component["path"],
                "owner_name": component["name"],
                "owner_class": component["nx_class"],
                "name": name,
                "transform": transform,
            }

    return index


def resolve_depends_on_reference(
    depends_on,
    owner_component,
    transform_index,
):
    """
    Resolve a depends_on reference to a transform-index key.

    Handles:
      - `.`
      - absolute NeXus paths
      - local bare transform names
      - `transformations/name`
      - relative component paths where possible

    The final basename fallback is accepted only if it produces
    exactly one unambiguous transformation match.
    """
    if depends_on in {None, "", "."}:
        return None

    depends_on = str(depends_on).strip()

    candidates = []

    # Full NeXus absolute path.
    if depends_on.startswith("/"):
        candidates.append(normalise_nexus_path(depends_on))

    else:
        # Local bare transform name: "location".
        if "/" not in depends_on:
            candidates.append(
                transform_absolute_path(
                    owner_component,
                    depends_on,
                )
            )

        # Relative path below the component, for example:
        # transformations/location
        candidates.append(
            normalise_nexus_path(
                f"{owner_component['path']}/{depends_on}"
            )
        )

        # Relative path from the component parent.
        parent_path = owner_component["path"].rsplit("/", 1)[0]

        if parent_path:
            candidates.append(
                normalise_nexus_path(
                    f"{parent_path}/{depends_on}"
                )
            )

    for candidate in candidates:
        if candidate in transform_index:
            return candidate

    # If an absolute path may have a different root prefix,
    # try an exact suffix match. This remains safe only when
    # exactly one transform matches.
    suffix = depends_on.strip("/")

    suffix_matches = [
        path
        for path in transform_index
        if path.strip("/").endswith(suffix)
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # Last fallback: transform basename, only if unique.
    basename = depends_on.rstrip("/").split("/")[-1]

    basename_matches = [
        path
        for path, entry in transform_index.items()
        if entry["name"] == basename
    ]

    if len(basename_matches) == 1:
        return basename_matches[0]

    return None


def component_dependency_chain(
    component,
    transform_index,
):
    """
    Return the selected component's chain as indexed transforms.

    Entries are returned from terminal transformation upward through
    `depends_on`, including transforms owned by other components.
    """
    transforms = collect_transformations(component)

    terminal_name = component_terminal_transform(
        component,
        transforms,
    )

    if terminal_name is None:
        return []

    current_path = transform_absolute_path(
        component,
        terminal_name,
    )

    result = []
    visited = set()

    while current_path and current_path not in visited:
        visited.add(current_path)

        entry = transform_index.get(current_path)

        if entry is None:
            result.append({
                "missing": True,
                "path": current_path,
                "name": current_path.split("/")[-1],
            })
            break

        result.append({
            "missing": False,
            "path": entry["path"],
            "owner_path": entry["owner_path"],
            "owner_name": entry["owner_name"],
            "owner_class": entry["owner_class"],
            "name": entry["name"],
            "type": entry["transform"].get("type"),
            "value": entry["transform"].get("value"),
            "vector": entry["transform"].get("vector"),
            "units": entry["transform"].get("units"),
            "depends_on": entry["transform"].get("depends_on"),
        })

        owner_component = {
            "path": entry["owner_path"],
        }

        # Build a minimal owner representation for local reference
        # resolution. Full component data is not needed here.
        owner_component = next(
            (
                candidate
                for candidate in _dependency_components_cache
                if candidate["path"] == entry["owner_path"]
            ),
            None,
        )

        if owner_component is None:
            break

        current_path = resolve_depends_on_reference(
            entry["transform"].get("depends_on"),
            owner_component,
            transform_index,
        )

    return result


# Cached only during dependency calculations in one callback.
_dependency_components_cache = []


def dependent_components(selected_component, all_components):
    """
    Return only components genuinely dependent on selected_component.

    A component is included if its resolved depends_on chain contains
    one of the selected component's transformation paths.

    The selected component itself is excluded.
    """
    global _dependency_components_cache

    _dependency_components_cache = all_components

    transform_index = build_transform_index(all_components)

    selected_transform_paths = {
        transform_absolute_path(selected_component, name)
        for name in collect_transformations(selected_component)
    }

    if not selected_transform_paths:
        return []

    result = []

    for candidate in all_components:
        if candidate["path"] == selected_component["path"]:
            continue

        chain = component_dependency_chain(
            candidate,
            transform_index,
        )

        candidate_paths = {
            entry["path"]
            for entry in chain
            if not entry.get("missing")
        }

        if candidate_paths & selected_transform_paths:
            result.append(candidate)

    return result


def dependency_panel_content(data, selected_component_path):
    """Render dependency information for the right panel."""
    global _dependency_components_cache

    base_style = {
        "fontFamily": "monospace",
        "fontSize": "12px",
    }

    if not selected_component_path:
        return html.Div(
            "Double-click a 3D point to inspect dependencies.",
            style={
                **base_style,
                "fontStyle": "italic",
                "color": "#666",
            },
        )

    components = collect_components(data)
    _dependency_components_cache = components

    selected_component = next(
        (
            component
            for component in components
            if component["path"] == selected_component_path
        ),
        None,
    )

    if selected_component is None:
        return html.Div(
            "Selected component is no longer available.",
            style={
                **base_style,
                "color": "#aa0000",
            },
        )

    transform_index = build_transform_index(components)

    chain = component_dependency_chain(
        selected_component,
        transform_index,
    )

    dependents = dependent_components(
        selected_component,
        components,
    )

    if not chain:
        chain_content = [
            html.Div(
                "No transformation chain found.",
                style={
                    "fontStyle": "italic",
                    "color": "#666",
                },
            )
        ]

    else:
        chain_content = []

        for index, entry in enumerate(chain):
            prefix = "└─ " if index else ""

            if entry.get("missing"):
                chain_content.append(
                    html.Div(
                        f"{prefix}⚠ {entry['name']} (unresolved)",
                        style={
                            "marginLeft": f"{index * 10}px",
                            "color": "#aa0000",
                        },
                    )
                )
                continue

            chain_content.append(
                html.Div(
                    [
                        html.Div(
                            (
                                f"{prefix}{entry['name']} "
                                f"[{entry['owner_name']}]"
                            ),
                            style={"fontWeight": "600"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    f"owner: {entry['owner_path']}"
                                ),
                                html.Div(
                                    f"type: {entry.get('type')}"
                                ),
                                html.Div(
                                    f"value: {entry.get('value')}"
                                ),
                                html.Div(
                                    f"vector: {entry.get('vector')}"
                                ),
                                html.Div(
                                    f"units: {entry.get('units')}"
                                ),
                                html.Div(
                                    "depends_on: "
                                    f"{entry.get('depends_on')}"
                                ),
                            ],
                            style={
                                "paddingLeft": "14px",
                                "color": "#555",
                            },
                        ),
                    ],
                    style={
                        "marginLeft": f"{index * 10}px",
                        "marginBottom": "8px",
                    },
                )
            )

    if not dependents:
        dependent_content = [
            html.Div(
                "No components depend on the selected component.",
                style={
                    "fontStyle": "italic",
                    "color": "#666",
                },
            )
        ]

    else:
        dependent_content = [
            html.Div(
                [
                    html.Div(
                        (
                            f"• {component['name']} "
                            f"[{component['nx_class']}]"
                        ),
                        style={"fontWeight": "600"},
                    ),
                    html.Div(
                        component["path"],
                        style={
                            "paddingLeft": "12px",
                            "fontSize": "11px",
                            "color": "#666",
                            "wordBreak": "break-word",
                        },
                    ),
                ],
                style={"marginBottom": "8px"},
            )
            for component in dependents
        ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        (
                            f"{selected_component['name']} "
                            f"[{selected_component['nx_class']}]"
                        ),
                        style={
                            "fontWeight": "bold",
                            "fontSize": "13px",
                        },
                    ),
                    html.Div(
                        selected_component["path"],
                        style={
                            "fontSize": "11px",
                            "color": "#666",
                            "wordBreak": "break-word",
                        },
                    ),
                ],
                style={"marginBottom": "14px"},
            ),

            html.Div(
                "Depends on",
                style={
                    "fontWeight": "bold",
                    "fontSize": "13px",
                    "borderBottom": "1px solid #bbb",
                    "paddingBottom": "3px",
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                chain_content,
                style={"marginBottom": "22px"},
            ),

            html.Div(
                "Components depending on selected component",
                style={
                    "fontWeight": "bold",
                    "fontSize": "13px",
                    "borderBottom": "1px solid #bbb",
                    "paddingBottom": "3px",
                    "marginBottom": "8px",
                },
            ),
            html.Div(dependent_content),
        ],
        style=base_style,
    )


# =========================================================
# Lazy JSON tree rendering
# =========================================================
def component_tree_style(selected=False):
    style = {
        "padding": "2px 4px",
        "margin": "1px 0",
        "borderRadius": "3px",
        "border": "2px solid transparent",
    }

    if selected:
        style.update({
            "backgroundColor": "#fff3b0",
            "border": "2px solid #e67e22",
        })

    return style


def array_summary(value):
    if len(value) == 0:
        return "empty list"

    first = value[0]

    if isinstance(first, (int, float)):
        return f"numeric array, {len(value):,} values"

    if isinstance(first, str):
        return f"string array, {len(value):,} values"

    return f"list, {len(value):,} items"


def editable_input(name, value, json_path):
    """Render editable scalar values and small lists/objects."""
    if isinstance(value, bool):
        raw_value = "true" if value else "false"
        value_type = "boolean"

    elif value is None:
        raw_value = "null"
        value_type = "null"

    elif isinstance(value, (int, float, np.integer, np.floating)):
        raw_value = str(value)
        value_type = "number"

    elif isinstance(value, (list, dict)):
        raw_value = json.dumps(value)
        value_type = "JSON"

    else:
        raw_value = str(value)
        value_type = "string"

    input_properties = {
        "id": {
            "type": "json-edit",
            "path": path_key(json_path),
        },
        "value": raw_value,
        "debounce": True,
        "style": {
            "width": "170px",
            "fontFamily": "monospace",
            "fontSize": "12px",
        },
    }

    if (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
    ):
        input_properties["type"] = "number"
        input_properties["step"] = "any"
    else:
        input_properties["type"] = "text"

    return html.Div(
        [
            html.Span(
                f"{name}: ",
                style={
                    "fontWeight": "600",
                    "fontFamily": "monospace",
                },
            ),
            dcc.Input(**input_properties),
            html.Span(
                f" ({value_type})",
                style={
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                    "color": "#777",
                },
            ),
        ],
        style={
            "padding": "2px 0",
            "whiteSpace": "nowrap",
        },
    )


def lazy_json_tree_node(
    name,
    value,
    json_path,
    component_by_json_path,
    hidden_paths,
    open_paths,
    selected_component_path=None,
):
    """
    Render a JSON node lazily.

    Only child nodes of currently open branches are created.
    """
    current_path_key = path_key(json_path)
    is_open = current_path_key in open_paths

    component = component_by_json_path.get(
        current_path_key
    )

    base_style = {
        "fontFamily": "monospace",
        "fontSize": "13px",
        "padding": "2px 0",
    }

    toggle_style = {
        "border": "none",
        "background": "transparent",
        "cursor": "pointer",
        "padding": "0 5px 0 0",
        "fontSize": "12px",
        "width": "24px",
    }

    children_style = {
        "paddingLeft": "18px",
        "marginLeft": "6px",
        "borderLeft": "1px solid #d0d0d0",
    }

    leaf_style = {
        "fontFamily": "monospace",
        "fontSize": "13px",
        "padding": "2px 0",
        "whiteSpace": "nowrap",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    }

    if isinstance(value, dict):
        node_style = dict(base_style)

        if (
            component is not None
            and component["path"] == selected_component_path
        ):
            node_style.update(
                component_tree_style(selected=True)
            )

        header = [
            html.Button(
                "▼" if is_open else "▶",
                id={
                    "type": "json-tree-toggle",
                    "path": current_path_key,
                },
                n_clicks=0,
                style=toggle_style,
            )
        ]

        if component is not None:
            component_path = component["path"]

            header.extend([
                dcc.Checklist(
                    id={
                        "type": "component-visibility",
                        "path": component_path,
                    },
                    options=[
                        {
                            "label": (
                                f"{name} "
                                f"[{component['nx_class']}]"
                            ),
                            "value": component_path,
                        }
                    ],
                    value=(
                        []
                        if component_path in hidden_paths
                        else [component_path]
                    ),
                    inline=True,
                    style={
                        "display": "inline-block",
                        "fontFamily": "monospace",
                        "fontSize": "13px",
                        "fontWeight": "600",
                    },
                    inputStyle={
                        "marginRight": "5px",
                    },
                ),
                html.Span(
                    f" {{ {len(value)} }}",
                    style={"color": "#666"},
                ),
            ])

        else:
            header.append(
                html.Span(
                    f"{name} {{ {len(value)} }}",
                    style={"fontWeight": "600"},
                )
            )

        children = []

        if is_open:
            visible_items = list(value.items())[:MAX_TREE_ITEMS]

            children = [
                lazy_json_tree_node(
                    name=str(child_name),
                    value=child_value,
                    json_path=json_path + [child_name],
                    component_by_json_path=component_by_json_path,
                    hidden_paths=hidden_paths,
                    open_paths=open_paths,
                    selected_component_path=selected_component_path,
                )
                for child_name, child_value in visible_items
            ]

            if len(value) > MAX_TREE_ITEMS:
                children.append(
                    html.Div(
                        f"... {len(value) - MAX_TREE_ITEMS} more keys",
                        style=leaf_style,
                    )
                )

        return html.Div(
            [
                html.Div(header, style=node_style),
                html.Div(children, style=children_style)
                if is_open
                else None,
            ]
        )

    if isinstance(value, list):
        structural_lists = {"children", "attributes"}

        if name not in structural_lists:
            if len(value) <= MAX_EDITABLE_LIST_ITEMS:
                return editable_input(
                    name,
                    value,
                    json_path,
                )

            return html.Div(
                f"{name}: [{array_summary(value)}]",
                style=leaf_style,
            )

        children = []

        if is_open:
            visible_items = value[:MAX_TREE_ITEMS]

            children = [
                lazy_json_tree_node(
                    name=f"[{index}]",
                    value=item,
                    json_path=json_path + [index],
                    component_by_json_path=component_by_json_path,
                    hidden_paths=hidden_paths,
                    open_paths=open_paths,
                    selected_component_path=selected_component_path,
                )
                for index, item in enumerate(visible_items)
            ]

            if len(value) > MAX_TREE_ITEMS:
                children.append(
                    html.Div(
                        f"... {len(value) - MAX_TREE_ITEMS} more items",
                        style=leaf_style,
                    )
                )

        return html.Div(
            [
                html.Div(
                    [
                        html.Button(
                            "▼" if is_open else "▶",
                            id={
                                "type": "json-tree-toggle",
                                "path": current_path_key,
                            },
                            n_clicks=0,
                            style=toggle_style,
                        ),
                        html.Span(
                            f"{name} [ {len(value)} ]",
                            style={
                                "fontWeight": "600",
                                "fontFamily": "monospace",
                            },
                        ),
                    ],
                    style=base_style,
                ),
                html.Div(children, style=children_style)
                if is_open
                else None,
            ]
        )

    return editable_input(name, value, json_path)


# =========================================================
# Dash application
# =========================================================
def show_instrument_viewer(
    figure,
    data,
    components,
    filename,
    initial_exclude_patterns,
    max_points,
    limits,
    port=8050,
):
    app = Dash(__name__)
    app.title = f"NeXus Viewer - {filename}"

    initial_hidden_paths = {
        component["path"]
        for component in components
        if component_matches_patterns(
            component,
            initial_exclude_patterns,
        )
    }

    initial_visible_paths = {
        component["path"]
        for component in components
        if component["path"] not in initial_hidden_paths
    }

    trace_path_map = {
        str(trace_index): component["path"]
        for component in components
        for trace_index in component["trace_indices"]
    }

    def build_tree(
        tree_data,
        hidden_paths,
        open_paths,
        selected_component_path=None,
    ):
        current_components = collect_components(tree_data)

        component_by_json_path = {
            path_key(component["json_path"]): component
            for component in current_components
        }

        return lazy_json_tree_node(
            name="root",
            value=tree_data,
            json_path=[],
            component_by_json_path=component_by_json_path,
            hidden_paths=set(hidden_paths),
            open_paths=set(open_paths),
            selected_component_path=selected_component_path,
        )

    def number_input(
        component_id,
        value,
        placeholder,
        margin_right=False,
    ):
        style = {
            "width": "48%",
            "boxSizing": "border-box",
            "fontFamily": "monospace",
        }

        if margin_right:
            style["marginRight"] = "4%"

        return dcc.Input(
            id=component_id,
            type="number",
            value=value,
            placeholder=placeholder,
            step="any",
            style=style,
        )

    dependency_panel_default_style = {
        "display": "none",
        "width": "350px",
        "minWidth": "260px",
        "maxWidth": "520px",
        "height": "100vh",
        "overflow": "auto",
        "padding": "15px",
        "boxSizing": "border-box",
        "borderLeft": "3px solid #888",
        "backgroundColor": "#fcfcfc",
        "flex": "0 0 auto",
    }

    app.layout = html.Div(
        [
            dcc.Store(
                id="json-data-store",
                data=data,
            ),
            dcc.Store(
                id="undo-history-store",
                data=[],
            ),
            dcc.Store(
                id="redo-history-store",
                data=[],
            ),
            dcc.Store(
                id="visible-component-paths",
                data=list(initial_visible_paths),
            ),
            dcc.Store(
                id="trace-path-map",
                data=trace_path_map,
            ),
            dcc.Store(
                id="spatial-limits-store",
                data=limits,
            ),
            dcc.Store(
                id="selected-component-path",
                data=None,
            ),
            dcc.Store(
                id="open-json-paths",
                data=["[]"],
            ),
            dcc.Store(
                id="plot-click-state",
                data={
                    "path": None,
                    "time": 0.0,
                },
            ),
            dcc.Store(
                id="plot-camera-store",
                data={
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.5},
                    "up": {"x": 0, "y": 0, "z": 1},
                    "center": {"x": 0, "y": 0, "z": 0},
                },
            ),
            dcc.Store(
                id="dependency-panel-hidden",
                data=False,
            ),
            dcc.Download(id="download-json"),

            # -------------------------------------------------
            # Left JSON / editing panel
            # -------------------------------------------------
            html.Div(
                [
                    html.H3(
                        "JSON structure",
                        style={
                            "marginTop": "0",
                            "marginBottom": "8px",
                        },
                    ),

                    html.Div(
                        [
                            html.Button(
                                "Update geometry",
                                id="update-geometry",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Button(
                                "Undo",
                                id="undo-json-edit",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Button(
                                "Redo",
                                id="redo-json-edit",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Button(
                                "Save JSON...",
                                id="save-json",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "12px",
                        },
                    ),

                    html.Div(
                        "Spatial limits [m]",
                        style={
                            "fontWeight": "bold",
                            "fontSize": "13px",
                        },
                    ),

                    html.Div(
                        [
                            number_input(
                                "gui-xmin",
                                limits["xmin"],
                                "xmin",
                                margin_right=True,
                            ),
                            number_input(
                                "gui-xmax",
                                limits["xmax"],
                                "xmax",
                            ),
                        ],
                        style={"marginBottom": "5px"},
                    ),

                    html.Div(
                        [
                            number_input(
                                "gui-ymin",
                                limits["ymin"],
                                "ymin",
                                margin_right=True,
                            ),
                            number_input(
                                "gui-ymax",
                                limits["ymax"],
                                "ymax",
                            ),
                        ],
                        style={"marginBottom": "5px"},
                    ),

                    html.Div(
                        [
                            number_input(
                                "gui-zmin",
                                limits["zmin"],
                                "zmin",
                                margin_right=True,
                            ),
                            number_input(
                                "gui-zmax",
                                limits["zmax"],
                                "zmax",
                            ),
                        ],
                        style={"marginBottom": "6px"},
                    ),

                    html.Button(
                        "Apply limits",
                        id="apply-spatial-limits",
                        n_clicks=0,
                        style={
                            "cursor": "pointer",
                            "marginBottom": "12px",
                        },
                    ),

                    html.Div(
                        "Exclude from 3D view",
                        style={
                            "fontWeight": "bold",
                            "fontSize": "13px",
                        },
                    ),

                    dcc.Input(
                        id="exclude-patterns",
                        type="text",
                        value=", ".join(
                            initial_exclude_patterns
                        ),
                        style={
                            "width": "100%",
                            "boxSizing": "border-box",
                            "marginBottom": "6px",
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                        },
                    ),

                    html.Div(
                        [
                            html.Button(
                                "Apply exclusions",
                                id="apply-exclusions",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Button(
                                "Hide all",
                                id="hide-all-components",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Button(
                                "Show all",
                                id="show-all-components",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Button(
                                "Reset",
                                id="reset-default-exclusions",
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "12px",
                        },
                    ),

                    html.Div(
                        (
                            "Only open JSON branches are loaded. "
                            "Double-click a 3D point to open and "
                            "highlight its matching component."
                        ),
                        style={
                            "fontSize": "12px",
                            "color": "#666",
                            "marginBottom": "10px",
                        },
                    ),

                    html.Div(
                        id="json-tree",
                        children=build_tree(
                            tree_data=data,
                            hidden_paths=initial_hidden_paths,
                            open_paths={"[]"},
                        ),
                    ),
                ],
                style={
                    "width": "390px",
                    "resize": "horizontal",
                    "overflow": "auto",
                    "minWidth": "260px",
                    "maxWidth": "850px",
                    "flex": "0 0 auto",
                    "height": "100vh",
                    "padding": "15px",
                    "boxSizing": "border-box",
                    "borderRight": "3px solid #888",
                    "backgroundColor": "#fafafa",
                },
            ),

            # -------------------------------------------------
            # Middle 3D plot
            # -------------------------------------------------
            html.Div(
                [
                    html.Button(
                        "Show dependencies",
                        id="show-dependencies",
                        n_clicks=0,
                        style={
                            "display": "none",
                            "position": "absolute",
                            "right": "8px",
                            "top": "8px",
                            "zIndex": 10,
                            "cursor": "pointer",
                        },
                    ),

                    dcc.Graph(
                        id="instrument-geometry",
                        figure=figure,
                        config={
                            "responsive": True,
                            "displaylogo": False,

                            # Our clickData handler uses two clicks to select a component.
                            # Disable Plotly's native double-click camera reset.
                            "doubleClick": False,
                        },
                        style={
                            "height": "100vh",
                            "width": "100%",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "flex": "1 1 0",
                    "minWidth": "0",
                    "height": "100vh",
                },
            ),

            # -------------------------------------------------
            # Right dependency panel
            # -------------------------------------------------
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Dependencies",
                                style={
                                    "display": "inline-block",
                                    "marginTop": "0",
                                },
                            ),
                            html.Button(
                                "Hide",
                                id="hide-dependencies",
                                n_clicks=0,
                                style={
                                    "float": "right",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "12px",
                        },
                    ),

                    html.Div(
                        id="dependency-panel-content",
                        children=dependency_panel_content(
                            data,
                            None,
                        ),
                    ),
                ],
                id="dependency-panel",
                style=dependency_panel_default_style,
            ),
        ],
        style={
            "display": "flex",
            "height": "100vh",
            "width": "100vw",
            "margin": "0",
            "padding": "0",
            "overflow": "hidden",
        },
    )

    # ---------------------------------------------------------
    # Save modified JSON
    # ---------------------------------------------------------
    @app.callback(
        Output("download-json", "data"),
        Input("save-json", "n_clicks"),
        State("json-data-store", "data"),
        prevent_initial_call=True,
    )
    def save_json(_clicks, current_data):
        return dcc.send_string(
            json.dumps(
                current_data,
                indent=2,
                ensure_ascii=False,
            ),
            filename="modified_nexus.json",
        )

    # ---------------------------------------------------------
    # Remember the live 3D camera (orientation and zoom).
    # relayoutData can report either a whole camera object or
    # individual nested camera fields, so merge both forms.
    # ---------------------------------------------------------
    @app.callback(
        Output("plot-camera-store", "data"),
        Input("instrument-geometry", "relayoutData"),
        State("plot-camera-store", "data"),
        prevent_initial_call=True,
    )
    def remember_plot_camera(relayout_data, saved_camera):
        if not isinstance(relayout_data, dict):
            return no_update

        camera = copy.deepcopy(saved_camera or {})
        changed = False

        full_camera = relayout_data.get("scene.camera")
        if isinstance(full_camera, dict):
            for key in ("eye", "up", "center", "projection"):
                if key in full_camera:
                    camera[key] = copy.deepcopy(full_camera[key])
            changed = True

        for key, value in relayout_data.items():
            prefix = "scene.camera."
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].split(".")
            if len(parts) not in (2, 3):
                continue
            section = parts[0]
            if section not in ("eye", "up", "center", "projection"):
                continue
            target = camera.setdefault(section, {})
            for part in parts[1:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = copy.deepcopy(value)
            changed = True

        return camera if changed else no_update

    # ---------------------------------------------------------
    # Tree expand/collapse and Plotly double-click selection
    # ---------------------------------------------------------
    @app.callback(
        Output("open-json-paths", "data"),
        Output("selected-component-path", "data"),
        Output("plot-click-state", "data"),

        Input(
            {
                "type": "json-tree-toggle",
                "path": ALL,
            },
            "n_clicks",
        ),
        Input("instrument-geometry", "clickData"),

        State("open-json-paths", "data"),
        State("plot-click-state", "data"),
        State("trace-path-map", "data"),
        State("json-data-store", "data"),

        prevent_initial_call=True,
    )
    def open_tree_branch_or_select_plot_point(
        _toggle_clicks,
        click_data,
        open_paths,
        click_state,
        current_trace_path_map,
        current_data,
    ):
        triggered = ctx.triggered_id
        open_paths = set(open_paths or [])
        open_paths.add("[]")

        if click_state is None:
            click_state = {
                "path": None,
                "time": 0.0,
            }

        # Tree branch expand/collapse.
        if (
            isinstance(triggered, dict)
            and triggered.get("type") == "json-tree-toggle"
        ):
            clicked_path_key = triggered["path"]

            if clicked_path_key in open_paths:
                if clicked_path_key != "[]":
                    open_paths.remove(clicked_path_key)
            else:
                open_paths.add(clicked_path_key)

            return (
                list(open_paths),
                no_update,
                no_update,
            )

        # Plot single/double click.
        if triggered == "instrument-geometry":
            if not click_data or not click_data.get("points"):
                return no_update, no_update, no_update

            curve_number = click_data["points"][0].get(
                "curveNumber"
            )

            if curve_number is None:
                return no_update, no_update, no_update

            clicked_component_path = current_trace_path_map.get(
                str(curve_number)
            )

            if clicked_component_path is None:
                return no_update, no_update, no_update

            now = time.monotonic()

            is_double_click = (
                clicked_component_path == click_state.get("path")
                and now - click_state.get("time", 0.0)
                <= DOUBLE_CLICK_SECONDS
            )

            new_click_state = {
                "path": clicked_component_path,
                "time": now,
            }

            # A single click has no visible effect.
            if not is_double_click:
                return (
                    no_update,
                    no_update,
                    new_click_state,
                )

            selected_component = next(
                (
                    component
                    for component in collect_components(current_data)
                    if component["path"] == clicked_component_path
                ),
                None,
            )

            if selected_component is None:
                return (
                    no_update,
                    no_update,
                    new_click_state,
                )

            open_paths.update(
                parent_path_keys(
                    selected_component["json_path"]
                )
            )

            return (
                list(open_paths),
                clicked_component_path,
                new_click_state,
            )

        return no_update, no_update, no_update

    # ---------------------------------------------------------
    # Render only active JSON tree branches
    # ---------------------------------------------------------
    @app.callback(
        Output("json-tree", "children"),
        Input("json-data-store", "data"),
        Input("open-json-paths", "data"),
        Input("selected-component-path", "data"),
        Input("visible-component-paths", "data"),
    )
    def render_json_tree(
        current_data,
        open_paths,
        selected_component_path,
        visible_paths,
    ):
        components_now = collect_components(current_data)

        all_paths = {
            component["path"]
            for component in components_now
        }

        hidden_paths = all_paths - set(visible_paths or [])

        return build_tree(
            tree_data=current_data,
            hidden_paths=hidden_paths,
            open_paths=set(open_paths or ["[]"]),
            selected_component_path=selected_component_path,
        )

    # ---------------------------------------------------------
    # Dependency panel content
    # ---------------------------------------------------------
    @app.callback(
        Output("dependency-panel-content", "children"),
        Input("selected-component-path", "data"),
        Input("json-data-store", "data"),
    )
    def update_dependency_panel_content(
        selected_component_path,
        current_data,
    ):
        return dependency_panel_content(
            current_data,
            selected_component_path,
        )

    # ---------------------------------------------------------
    # Show/hide dependency panel
    # ---------------------------------------------------------
    @app.callback(
        Output("dependency-panel-hidden", "data"),
        Input("hide-dependencies", "n_clicks"),
        Input("show-dependencies", "n_clicks"),
        State("dependency-panel-hidden", "data"),
        prevent_initial_call=True,
    )
    def toggle_dependency_panel(
        _hide_clicks,
        _show_clicks,
        hidden,
    ):
        if ctx.triggered_id == "hide-dependencies":
            return True

        if ctx.triggered_id == "show-dependencies":
            return False

        return hidden

    @app.callback(
        Output("dependency-panel", "style"),
        Output("show-dependencies", "style"),
        Input("selected-component-path", "data"),
        Input("dependency-panel-hidden", "data"),
    )
    def update_dependency_panel_visibility(
        selected_path,
        hidden,
    ):
        panel_style = {
            "width": "350px",
            "minWidth": "260px",
            "maxWidth": "520px",
            "height": "100vh",
            "overflow": "auto",
            "padding": "15px",
            "boxSizing": "border-box",
            "borderLeft": "3px solid #888",
            "backgroundColor": "#fcfcfc",
            "flex": "0 0 auto",
        }

        show_button_style = {
            "position": "absolute",
            "right": "8px",
            "top": "8px",
            "zIndex": 10,
            "cursor": "pointer",
        }

        # No selected component: do not show a dependency panel.
        if not selected_path:
            panel_style["display"] = "none"
            show_button_style["display"] = "none"

            return panel_style, show_button_style

        # Folded away by the user.
        if hidden:
            panel_style["display"] = "none"
            show_button_style["display"] = "block"

            return panel_style, show_button_style

        panel_style["display"] = "block"
        show_button_style["display"] = "none"

        return panel_style, show_button_style

    # ---------------------------------------------------------
    # Main state updates:
    # editing, undo/redo, visibility and spatial limits
    # ---------------------------------------------------------
    @app.callback(
        Output("instrument-geometry", "figure"),
        Output("json-data-store", "data"),
        Output("undo-history-store", "data"),
        Output("redo-history-store", "data"),
        Output("visible-component-paths", "data"),
        Output("trace-path-map", "data"),
        Output("spatial-limits-store", "data"),
        Output("exclude-patterns", "value"),

        Input(
            {
                "type": "component-visibility",
                "path": ALL,
            },
            "value",
        ),
        Input(
            {
                "type": "json-edit",
                "path": ALL,
            },
            "value",
        ),
        Input("apply-exclusions", "n_clicks"),
        Input("hide-all-components", "n_clicks"),
        Input("show-all-components", "n_clicks"),
        Input("reset-default-exclusions", "n_clicks"),
        Input("update-geometry", "n_clicks"),
        Input("apply-spatial-limits", "n_clicks"),
        Input("undo-json-edit", "n_clicks"),
        Input("redo-json-edit", "n_clicks"),
        # Selection expands the side panel and can resize the graph.
        # Reapply the last user camera after that layout change.
        Input("selected-component-path", "data"),

        State("exclude-patterns", "value"),
        State(
            {
                "type": "component-visibility",
                "path": ALL,
            },
            "id",
        ),
        State(
            {
                "type": "json-edit",
                "path": ALL,
            },
            "id",
        ),
        State("gui-xmin", "value"),
        State("gui-xmax", "value"),
        State("gui-ymin", "value"),
        State("gui-ymax", "value"),
        State("gui-zmin", "value"),
        State("gui-zmax", "value"),
        State("spatial-limits-store", "data"),
        State("json-data-store", "data"),
        State("undo-history-store", "data"),
        State("redo-history-store", "data"),
        State("visible-component-paths", "data"),
        State("instrument-geometry", "figure"),
        State("trace-path-map", "data"),
        State("plot-camera-store", "data"),

        prevent_initial_call=True,
    )
    def update_viewer_state(
        checkbox_values,
        edit_values,
        _apply_exclusions_clicks,
        _hide_all_clicks,
        _show_all_clicks,
        _reset_clicks,
        _update_geometry_clicks,
        _apply_limits_clicks,
        _undo_clicks,
        _redo_clicks,
        _selected_component_path,
        exclude_text,
        checkbox_ids,
        edit_ids,
        gui_xmin,
        gui_xmax,
        gui_ymin,
        gui_ymax,
        gui_zmin,
        gui_zmax,
        current_limits,
        current_data,
        undo_history,
        redo_history,
        current_visible_paths,
        current_figure,
        current_trace_path_map,
        saved_camera,
    ):
        triggered = ctx.triggered_id

        new_data = copy.deepcopy(current_data)
        undo_history = undo_history or []
        redo_history = redo_history or []
        active_limits = dict(current_limits or limits)

        exclude_patterns = parse_exclude_patterns(
            exclude_text
        )

        visible_paths = set(current_visible_paths or [])

        # -------------------------------------------------
        # Apply JSON edit
        # -------------------------------------------------
        if (
            isinstance(triggered, dict)
            and triggered.get("type") == "json-edit"
        ):
            old_data_snapshot = copy.deepcopy(current_data)
            changed = False

            for value, item_id in zip(edit_values, edit_ids):
                try:
                    json_path = json.loads(item_id["path"])

                    old_value = get_value_at_path(
                        new_data,
                        json_path,
                    )

                    new_value = parse_edited_value(
                        value,
                        old_value,
                    )

                    if new_value != old_value:
                        set_value_at_path(
                            new_data,
                            json_path,
                            new_value,
                        )
                        changed = True

                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    json.JSONDecodeError,
                ):
                    pass

            if changed:
                undo_history.append(old_data_snapshot)
                undo_history = undo_history[-MAX_HISTORY_LENGTH:]
                redo_history = []

        # -------------------------------------------------
        # Undo / redo
        # -------------------------------------------------
        if triggered == "undo-json-edit" and undo_history:
            redo_history.append(copy.deepcopy(current_data))
            redo_history = redo_history[-MAX_HISTORY_LENGTH:]
            new_data = undo_history.pop()

        elif triggered == "redo-json-edit" and redo_history:
            undo_history.append(copy.deepcopy(current_data))
            undo_history = undo_history[-MAX_HISTORY_LENGTH:]
            new_data = redo_history.pop()

        # -------------------------------------------------
        # Apply GUI spatial limits
        # -------------------------------------------------
        if triggered == "apply-spatial-limits":
            active_limits = validate_limits({
                "xmin": gui_xmin,
                "xmax": gui_xmax,
                "ymin": gui_ymin,
                "ymax": gui_ymax,
                "zmin": gui_zmin,
                "zmax": gui_zmax,
            })

        components_now = collect_components(new_data)

        component_paths = {
            component["path"]
            for component in components_now
        }

        component_by_path = {
            component["path"]: component
            for component in components_now
        }

        # -------------------------------------------------
        # Visibility controls
        # -------------------------------------------------
        if triggered == "hide-all-components":
            visible_paths = set()

        elif triggered == "show-all-components":
            visible_paths = set(component_paths)
            exclude_text = ""
            exclude_patterns = []

        elif triggered == "reset-default-exclusions":
            exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
            exclude_text = ", ".join(
                DEFAULT_EXCLUDE_PATTERNS
            )

            visible_paths = {
                component["path"]
                for component in components_now
                if not component_matches_patterns(
                    component,
                    exclude_patterns,
                )
            }

        elif triggered == "apply-exclusions":
            visible_paths = {
                component["path"]
                for component in components_now
                if not component_matches_patterns(
                    component,
                    exclude_patterns,
                )
            }

        elif (
            isinstance(triggered, dict)
            and triggered.get("type") == "component-visibility"
        ):
            changed_path = triggered["path"]
            changed_values = []

            for values, checkbox_id in zip(
                checkbox_values,
                checkbox_ids,
            ):
                if checkbox_id["path"] == changed_path:
                    changed_values = values or []
                    break

            component = component_by_path.get(changed_path)

            if component is not None:
                checked = changed_path in changed_values

                excluded = component_matches_patterns(
                    component,
                    exclude_patterns,
                )

                if checked and not excluded:
                    visible_paths.add(changed_path)
                else:
                    visible_paths.discard(changed_path)

        visible_paths &= component_paths

        # -------------------------------------------------
        # Rebuild geometry after explicit update, limits,
        # undo or redo.
        # -------------------------------------------------
        rebuild_geometry = (
            triggered == "update-geometry"
            or triggered == "apply-spatial-limits"
            or triggered == "undo-json-edit"
            or triggered == "redo-json-edit"
            or (
                isinstance(triggered, dict)
                and triggered.get("type") == "json-edit"
            )
        )

        if rebuild_geometry:
            new_figure, plotted_components = plot_instrument(
                data=new_data,
                json_file="modified JSON",
                exclude_patterns=exclude_patterns,
                visible_paths=visible_paths,
                xmin=active_limits["xmin"],
                xmax=active_limits["xmax"],
                ymin=active_limits["ymin"],
                ymax=active_limits["ymax"],
                zmin=active_limits["zmin"],
                zmax=active_limits["zmax"],
                max_points=max_points,
            )

            current_figure = new_figure.to_dict()

            current_trace_path_map = {
                str(trace_index): component["path"]
                for component in plotted_components
                for trace_index in component["trace_indices"]
            }

        # -------------------------------------------------
        # Restore the last known camera after a selection or
        # figure rebuild. Camera.eye controls 3D zoom; center/up
        # and eye together preserve orientation and target.
        # -------------------------------------------------
        if current_figure and isinstance(saved_camera, dict) and saved_camera:
            layout = current_figure.setdefault("layout", {})
            scene = layout.setdefault("scene", {})
            scene["camera"] = copy.deepcopy(saved_camera)
            # Keep stable UI revisions so Plotly does not discard
            # the camera on subsequent figure updates.
            layout["uirevision"] = PLOT_UIREVISION
            scene["uirevision"] = PLOT_SCENE_UIREVISION

        # -------------------------------------------------
        # Update existing trace visibility.
        # -------------------------------------------------
        if current_figure:
            for trace_index_text, component_path in (
                current_trace_path_map.items()
            ):
                trace_index = int(trace_index_text)

                if trace_index < len(current_figure["data"]):
                    current_figure["data"][trace_index][
                        "visible"
                    ] = component_path in visible_paths

        return (
            current_figure,
            new_data,
            undo_history,
            redo_history,
            list(visible_paths),
            current_trace_path_map,
            active_limits,
            exclude_text,
        )

    host = "127.0.0.1"
    url = f"http://{host}:{port}/"

    print(f"\nOpening viewer: {url}\n")

    Timer(
        1.0,
        lambda: webbrowser.open_new_tab(url),
    ).start()

    app.run(
        host=host,
        port=port,
        debug=False,
    )


# =========================================================
# Command-line interface
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Display and edit NeXus JSON geometry in a browser."
        )
    )

    parser.add_argument(
        "filename",
        help="Path to the NeXus JSON file",
    )

    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Additional initial wildcard exclusion patterns.",
    )

    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--zmin", type=float)
    parser.add_argument("--zmax", type=float)

    parser.add_argument(
        "--max-points",
        type=int,
        default=100_000,
        help=(
            "Maximum child geometry points per component. "
            "Default: 100000."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Local Dash server port. Default: 8050.",
    )

    args = parser.parse_args()

    with open(args.filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    limits = validate_limits({
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
        "zmin": args.zmin,
        "zmax": args.zmax,
    })

    initial_exclude_patterns = (
        DEFAULT_EXCLUDE_PATTERNS
        + args.exclude
    )

    figure, components = plot_instrument(
        data=data,
        json_file=args.filename,
        exclude_patterns=initial_exclude_patterns,
        xmin=limits["xmin"],
        xmax=limits["xmax"],
        ymin=limits["ymin"],
        ymax=limits["ymax"],
        zmin=limits["zmin"],
        zmax=limits["zmax"],
        max_points=args.max_points,
    )

    show_instrument_viewer(
        figure=figure,
        data=data,
        components=components,
        filename=args.filename,
        initial_exclude_patterns=initial_exclude_patterns,
        max_points=args.max_points,
        limits=limits,
        port=args.port,
    )


if __name__ == "__main__":
    main()