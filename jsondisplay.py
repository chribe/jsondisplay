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

# A fixed value preserves the Plotly 3D camera while Dash
# updates tree state, visibility, or JSON values.
PLOT_UIREVISION = "nexus-json-geometry-viewer-v1"


# =========================================================
# General utilities
# =========================================================
def attrs_to_dict(attrs):
    """Convert NeXus-style attributes into a normal dictionary."""
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
    """Generate a stable colour for each NX_class."""
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
    """
    Parse comma-, semicolon-, or newline-separated patterns.
    """
    if not text:
        return []

    text = text.replace(";", ",").replace("\n", ",")

    return [
        pattern.strip()
        for pattern in text.split(",")
        if pattern.strip()
    ]


def component_matches_patterns(component, patterns):
    """
    Match wildcard patterns against component name, NX_class,
    and full NeXus path. Matching is case-insensitive.
    """
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
    """Return a boolean mask selecting points inside spatial limits."""
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
    """Limit point clouds before sending them to Plotly."""
    point_count = len(point_set["x"])

    if max_points is None or point_count <= max_points:
        return point_set

    indices = np.linspace(
        0,
        point_count - 1,
        max_points,
        dtype=int,
    )

    print(
        f"Downsampling {point_set['name']}: "
        f"{point_count} -> {max_points} points"
    )

    return {
        "name": point_set["name"],
        "x": point_set["x"][indices],
        "y": point_set["y"][indices],
        "z": point_set["z"][indices],
    }


def path_key(json_path):
    """Return a stable serialisable key for a JSON path."""
    return json.dumps(json_path)


def parent_path_keys(json_path):
    """
    Return the root and all parent JSON paths for a node.

    Example:
        ["children", 0, "children", 2]

    returns keys for:
        []
        ["children"]
        ["children", 0]
        ["children", 0, "children"]
        ["children", 0, "children", 2]
    """
    return [
        path_key(json_path[:index])
        for index in range(len(json_path) + 1)
    ]


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
    Convert editable text back into JSON-compatible data.

    Numeric values can freely transition, for example:

        1.5 -> 10 -> 1.5
        3 -> 2.75 -> 4
        1e-3 -> 100 -> -0.5
    """
    text = str(text).strip()

    # Boolean
    if isinstance(old_value, bool):
        if text.lower() in {"true", "1", "yes", "on"}:
            return True

        if text.lower() in {"false", "0", "no", "off"}:
            return False

        raise ValueError(
            f"Expected boolean value, got {text!r}"
        )

    # Null
    if old_value is None:
        if text.lower() == "null":
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    # Number
    if (
        isinstance(old_value, (int, float, np.integer, np.floating))
        and not isinstance(old_value, bool)
    ):
        numeric_value = float(text)

        entered_as_integer = (
            "." not in text
            and "e" not in text.lower()
        )

        if entered_as_integer:
            return int(numeric_value)

        return numeric_value

    # JSON list / object
    if isinstance(old_value, (list, dict)):
        return json.loads(text)

    # String
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
    Collect all groups with an NX_class attribute.

    json_path stores the exact raw JSON location of the group.
    It is used to lazy-open the matching tree path after a
    double-click in the Plotly 3D view.
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


def collect_transformations(component):
    """Collect local transformation datasets for one component."""
    transforms = {}

    for child in component["node"].get("children", []):
        if (
            child.get("type") != "group"
            or child.get("name") != "transformations"
        ):
            continue

        for transform in child.get("children", []):
            if transform.get("module") != "dataset":
                continue

            config = transform.get("config", {})
            name = config.get("name")

            if name is None:
                continue

            attrs = attrs_to_dict(
                transform.get("attributes", [])
            )

            transforms[name] = {
                "value": config.get("values"),
                "type": attrs.get("transformation_type"),
                "vector": attrs.get("vector"),
                "units": attrs.get("units"),
                "depends_on": attrs.get("depends_on"),
            }

    return transforms


def transform_to_matrix(transform):
    """Convert one transformation dataset to a 4x4 matrix."""
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


def resolve_transform_chain(transforms, current_name, visited=None):
    """Resolve a local NeXus depends_on transformation chain."""
    if visited is None:
        visited = set()

    if current_name in {None, ".", ""}:
        return np.eye(4)

    if current_name in visited:
        print(
            f"WARNING: dependency loop at '{current_name}'."
        )
        return np.eye(4)

    visited.add(current_name)

    if current_name not in transforms:
        print(
            f"WARNING: transform '{current_name}' "
            "not found locally."
        )
        return np.eye(4)

    current = transforms[current_name]
    depends_on = current.get("depends_on")

    parent_name = None

    if depends_on:
        parent_name = str(depends_on).split("/")[-1]

    parent_matrix = resolve_transform_chain(
        transforms,
        parent_name,
        visited,
    )

    return parent_matrix @ transform_to_matrix(current)


def build_component_matrix(component):
    """Build the global 4x4 transformation matrix for a component."""
    transforms = collect_transformations(component)

    if not transforms:
        return np.eye(4)

    depends_on = None

    for child in component["node"].get("children", []):
        if child.get("module") != "dataset":
            continue

        config = child.get("config", {})

        if config.get("name") == "depends_on":
            depends_on = config.get("values")
            break

    if depends_on is not None:
        root_name = str(depends_on).split("/")[-1]

    else:
        referenced = {
            str(transform["depends_on"]).split("/")[-1]
            for transform in transforms.values()
            if transform.get("depends_on")
        }

        terminal_transforms = [
            name
            for name in transforms
            if name not in referenced
        ]

        if not terminal_transforms:
            return np.eye(4)

        root_name = terminal_transforms[0]

    return resolve_transform_chain(
        transforms,
        root_name,
    )


# =========================================================
# Geometry extraction
# =========================================================
def extract_child_points(component, matrix, limits=None):
    """Extract XYZ offset datasets and transform them globally."""
    datasets = {}
    point_sets = []

    for child in component["node"].get("children", []):
        if child.get("module") != "dataset":
            continue

        config = child.get("config", {})
        name = config.get("name")

        if name is not None:
            datasets[name] = config.get("values")

    xyz_candidates = [
        ("x_pixel_offset", "y_pixel_offset", "z_pixel_offset"),
        ("x_offset", "y_offset", "z_offset"),
    ]

    for x_name, y_name, z_name in xyz_candidates:
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
    """Create the Plotly geometry figure."""
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
    figure = go.Figure()

    # Used to create symmetric X/Y ranges, placing (0, 0)
    # in the centre of the initial view.
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

        matrix = build_component_matrix(component)
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

                # Store component path once per trace.
                meta={
                    "component_path": component["path"],
                },

                marker=dict(
                    size=8,
                    color=colour,
                    line=dict(
                        color="black",
                        width=1,
                    ),
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

                    # Stored once per trace, rather than once
                    # for every detector/pixel point.
                    meta={
                        "component_path": component["path"],
                    },

                    marker=dict(
                        size=1.5,
                        color=colour,
                        opacity=0.25,
                    ),

                    # Keeps points clickable while hiding hover text.
                    hoverinfo="none",
                )
            )

    # Ensure usable default ranges even for empty geometry.
    x_abs_max = max(x_abs_max, 1.0)
    y_abs_max = max(y_abs_max, 1.0)

    x_range = [-x_abs_max, x_abs_max]
    y_range = [-y_abs_max, y_abs_max]

    figure.update_layout(
        title=f"NeXus JSON Geometry: {json_file}",

        # Keep camera orientation, zoom, and pan while Dash
        # updates the figure in response to UI events.
        uirevision=PLOT_UIREVISION,

        scene=dict(
            xaxis=dict(
                title="X [m]",
                range=x_range,
            ),
            yaxis=dict(
                title="Y [m]",
                range=y_range,
            ),
            zaxis=dict(
                title="Z [m]",
            ),

            # Keep aspect ratios based on the data.
            aspectmode="data",

            # Preserve scene camera state after updates.
            uirevision=PLOT_UIREVISION,

            # Initial viewing orientation. The symmetric X/Y
            # ranges place global X=0 and Y=0 in the centre.
            camera=dict(
                eye=dict(
                    x=1.5,
                    y=1.5,
                    z=1.5,
                ),
                up=dict(
                    x=0,
                    y=0,
                    z=1,
                ),
                center=dict(
                    x=0,
                    y=0,
                    z=0,
                ),
            ),
        ),

        height=900,
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0,
        ),
        showlegend=True,
    )

    return figure, components


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
    """Render an editable scalar or a small JSON list/object."""
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
    Render one JSON tree node.

    Child nodes are only rendered when this node is open.
    This prevents construction of a very large hidden DOM tree.
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

    # -----------------------------------------------------
    # JSON object
    # -----------------------------------------------------
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

        # Lazy loading: build object children only when open.
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

    # -----------------------------------------------------
    # JSON list
    # -----------------------------------------------------
    if isinstance(value, list):
        structural_lists = {"children", "attributes"}

        # Data arrays are either editable when small or summarised.
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

        # Lazy loading: structural list entries only exist
        # after the list has been opened.
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

    # -----------------------------------------------------
    # JSON scalar
    # -----------------------------------------------------
    return editable_input(name, value, json_path)


# =========================================================
# Dash viewer
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

    trace_path_map = {}

    for component in components:
        for trace_index in component["trace_indices"]:
            trace_path_map[
                str(trace_index)
            ] = component["path"]

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
            dcc.Download(id="download-json"),

            # -------------------------------------------------
            # Left JSON tree / editing panel
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
                            "Click ▶ to open a branch. "
                            "Double-click a 3D point to open and "
                            "highlight its matching component. "
                            "Single plot clicks do nothing."
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
            # Right Plotly 3D panel
            # -------------------------------------------------
            html.Div(
                [
                    dcc.Graph(
                        id="instrument-geometry",
                        figure=figure,
                        config={
                            "responsive": True,
                            "displaylogo": False,

                            # Disable Plotly's normal double-click
                            # camera reset. Double-click is used for
                            # JSON-tree navigation instead.
                            "doubleClick": False,
                        },
                        style={
                            "height": "100vh",
                            "width": "100%",
                        },
                    ),
                ],
                style={
                    "flex": "1 1 0",
                    "minWidth": "0",
                    "height": "100vh",
                },
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
    # Tree expansion and plot double-click handling
    #
    # One callback updates open-json-paths, avoiding duplicate
    # Dash callback outputs.
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

        # -------------------------------------------------
        # Expand/collapse a tree node
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Plot single/double click
        # -------------------------------------------------
        if triggered == "instrument-geometry":
            if not click_data or not click_data.get("points"):
                return no_update, no_update, no_update

            point = click_data["points"][0]
            curve_number = point.get("curveNumber")

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

            # Single click changes nothing visibly.
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

            # Open root and every branch leading to the
            # selected component.
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
    # Render only currently open tree branches
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
        current_components = collect_components(current_data)

        all_paths = {
            component["path"]
            for component in current_components
        }

        hidden_paths = all_paths - set(visible_paths or [])

        return build_tree(
            tree_data=current_data,
            hidden_paths=hidden_paths,
            open_paths=set(open_paths or ["[]"]),
            selected_component_path=selected_component_path,
        )

    # ---------------------------------------------------------
    # Main state update: editing, visibility, undo/redo,
    # and explicit geometry rebuilding.
    # ---------------------------------------------------------
    @app.callback(
        Output("instrument-geometry", "figure"),
        Output("json-data-store", "data"),
        Output("undo-history-store", "data"),
        Output("redo-history-store", "data"),
        Output("visible-component-paths", "data"),
        Output("trace-path-map", "data"),
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
        Input("undo-json-edit", "n_clicks"),
        Input("redo-json-edit", "n_clicks"),

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
        State("json-data-store", "data"),
        State("undo-history-store", "data"),
        State("redo-history-store", "data"),
        State("visible-component-paths", "data"),
        State("instrument-geometry", "figure"),
        State("trace-path-map", "data"),

        prevent_initial_call=True,
    )
    def update_viewer_state(
        checkbox_values,
        edit_values,
        _apply_clicks,
        _hide_all_clicks,
        _show_all_clicks,
        _reset_clicks,
        _update_geometry_clicks,
        _undo_clicks,
        _redo_clicks,
        exclude_text,
        checkbox_ids,
        edit_ids,
        current_data,
        undo_history,
        redo_history,
        current_visible_paths,
        current_figure,
        current_trace_path_map,
    ):
        triggered = ctx.triggered_id

        new_data = copy.deepcopy(current_data)
        undo_history = undo_history or []
        redo_history = redo_history or []

        exclude_patterns = parse_exclude_patterns(
            exclude_text
        )

        current_components = collect_components(new_data)

        all_component_paths = {
            component["path"]
            for component in current_components
        }

        visible_paths = set(current_visible_paths or [])

        # -------------------------------------------------
        # Apply one JSON edit
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
                    # Invalid input is ignored until corrected.
                    pass

            if changed:
                undo_history.append(old_data_snapshot)
                undo_history = undo_history[-MAX_HISTORY_LENGTH:]

                # A new edit clears redo history.
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

        updated_components = collect_components(new_data)

        updated_component_paths = {
            component["path"]
            for component in updated_components
        }

        component_by_path = {
            component["path"]: component
            for component in updated_components
        }

        # -------------------------------------------------
        # Visibility handling
        # -------------------------------------------------
        if triggered == "hide-all-components":
            visible_paths = set()

        elif triggered == "show-all-components":
            visible_paths = set(updated_component_paths)
            exclude_text = ""
            exclude_patterns = []

        elif triggered == "reset-default-exclusions":
            exclude_patterns = DEFAULT_EXCLUDE_PATTERNS

            exclude_text = ", ".join(
                DEFAULT_EXCLUDE_PATTERNS
            )

            visible_paths = {
                component["path"]
                for component in updated_components
                if not component_matches_patterns(
                    component,
                    exclude_patterns,
                )
            }

        elif triggered == "apply-exclusions":
            visible_paths = {
                component["path"]
                for component in updated_components
                if not component_matches_patterns(
                    component,
                    exclude_patterns,
                )
            }

        elif (
            isinstance(triggered, dict)
            and triggered.get("type") == "component-visibility"
        ):
            # Update only the clicked checkbox. This is important
            # for lazy loading: unopened component checkboxes do not
            # exist in the browser and must retain their state.
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
                is_checked = changed_path in changed_values
                is_excluded = component_matches_patterns(
                    component,
                    exclude_patterns,
                )

                if is_checked and not is_excluded:
                    visible_paths.add(changed_path)
                else:
                    visible_paths.discard(changed_path)

        # Remove paths that no longer exist after edits/undo/redo.
        visible_paths &= updated_component_paths

        # -------------------------------------------------
        # Rebuild geometry only after explicit Update,
        # Undo, or Redo.
        # -------------------------------------------------
        rebuild_geometry = (
            triggered == "update-geometry"
            or triggered == "undo-json-edit"
            or triggered == "redo-json-edit"
        )

        if rebuild_geometry:
            new_figure, plotted_components = plot_instrument(
                data=new_data,
                json_file="modified JSON",
                exclude_patterns=exclude_patterns,
                visible_paths=visible_paths,
                xmin=limits["xmin"],
                xmax=limits["xmax"],
                ymin=limits["ymin"],
                ymax=limits["ymax"],
                zmin=limits["zmin"],
                zmax=limits["zmax"],
                max_points=max_points,
            )

            current_figure = new_figure.to_dict()

            current_trace_path_map = {}

            for component in plotted_components:
                for trace_index in component["trace_indices"]:
                    current_trace_path_map[
                        str(trace_index)
                    ] = component["path"]

        # -------------------------------------------------
        # Update visibility without rebuilding geometry.
        # The constant uirevision preserves the current camera.
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

    limits = {
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
        "zmin": args.zmin,
        "zmax": args.zmax,
    }

    initial_exclude_patterns = (
        DEFAULT_EXCLUDE_PATTERNS
        + args.exclude
    )

    figure, components = plot_instrument(
        data=data,
        json_file=args.filename,
        exclude_patterns=initial_exclude_patterns,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        zmin=args.zmin,
        zmax=args.zmax,
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