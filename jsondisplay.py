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
DOUBLE_CLICK_SECONDS = 0.40
MAX_HISTORY_LENGTH = 30


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
    """Generate a stable colour automatically for each NX_class."""
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
    """
    Match patterns case-insensitively against:
      - component name
      - NX_class
      - full NeXus path
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


def get_value_at_path(data, path):
    current = data

    for key in path:
        current = current[key]

    return current


def set_value_at_path(data, path, value):
    current = data

    for key in path[:-1]:
        current = current[key]

    current[path[-1]] = value


def parse_edited_value(text, old_value):
    """
    Parse edited JSON values.

    Numeric values may switch freely between integer and float:

        1.5 -> 10 -> 1.5
        2 -> 3.75 -> 4
        1e-3 -> 10 -> -0.25
    """
    text = str(text).strip()

    # Boolean
    if isinstance(old_value, bool):
        lower_text = text.lower()

        if lower_text in {"true", "1", "yes", "on"}:
            return True

        if lower_text in {"false", "0", "no", "off"}:
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

    # Numeric values
    if (
        isinstance(old_value, (int, float, np.integer, np.floating))
        and not isinstance(old_value, bool)
    ):
        try:
            numeric_value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expected numeric value, got {text!r}"
            ) from exc

        # Integer only if entered without decimal/exponent notation.
        entered_as_integer = (
            "." not in text
            and "e" not in text.lower()
        )

        if entered_as_integer:
            return int(numeric_value)

        return numeric_value

    # Lists and dictionaries must be valid JSON.
    if isinstance(old_value, (list, dict)):
        return json.loads(text)

    # Strings
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
# NeXus component collection
# =========================================================
def collect_components(
    node,
    path="",
    json_path=None,
    components=None,
):
    """
    Collect every node containing NX_class.

    json_path is used to expand the matching branch after a
    double-click in the 3D view.
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


# =========================================================
# NeXus transformations
# =========================================================
def collect_transformations(component):
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
    figure = go.Figure()

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

        colour = component_color(component["nx_class"])

        component["trace_indices"].append(
            len(figure.data)
        )

        figure.add_trace(
            go.Scatter3d(
                x=[x0],
                y=[y0],
                z=[z0],
                customdata=[component["path"]],
                mode="markers+text",
                name=component["name"],
                text=[component["name"]],
                textposition="top center",
                visible=visible,
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

        point_sets = extract_child_points(
            component,
            matrix,
            limits,
        )

        for point_set in point_sets:
            if len(point_set["x"]) == 0:
                continue

            point_set = downsample_points(
                point_set,
                max_points,
            )

            component["trace_indices"].append(
                len(figure.data)
            )

            figure.add_trace(
                go.Scatter3d(
                    x=point_set["x"],
                    y=point_set["y"],
                    z=point_set["z"],
                    customdata=np.full(
                        len(point_set["x"]),
                        component["path"],
                        dtype=object,
                    ),
                    mode="markers",
                    name=point_set["name"],
                    visible=visible,
                    marker=dict(
                        size=1.5,
                        color=colour,
                        opacity=0.25,
                    ),
                    hoverinfo="skip",
                )
            )

    figure.update_layout(
        title=f"NeXus JSON Geometry: {json_file}",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )

    return figure, components


# =========================================================
# JSON tree rendering
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
    if not value:
        return "empty list"

    if isinstance(value[0], (int, float)):
        return f"numeric array, {len(value):,} values"

    return f"list, {len(value):,} items"


def editable_input(name, value, json_path):
    """
    Render an editable scalar/small-list input.

    Numeric fields always use step='any', allowing transitions such as:
        1.5 -> 10 -> 1.5
    """
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
            "path": json.dumps(json_path),
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


def json_tree_node(
    name,
    value,
    json_path,
    component_lookup,
    hidden_paths,
    selected_path=None,
    selected_json_path=None,
    max_items=100,
    open_node=False,
):
    summary_style = {
        "fontWeight": "600",
        "fontFamily": "monospace",
        "fontSize": "13px",
        "cursor": "pointer",
        "padding": "2px 0",
    }

    children_style = {
        "paddingLeft": "14px",
        "marginLeft": "4px",
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

    open_selected_branch = (
        selected_json_path is not None
        and json_path == selected_json_path[:len(json_path)]
    )

    should_open = open_node or open_selected_branch

    # -----------------------------------------------------
    # JSON object
    # -----------------------------------------------------
    if isinstance(value, dict):
        component = component_lookup.get(id(value))
        items = list(value.items())

        children = [
            json_tree_node(
                name=str(key),
                value=child_value,
                json_path=json_path + [key],
                component_lookup=component_lookup,
                hidden_paths=hidden_paths,
                selected_path=selected_path,
                selected_json_path=selected_json_path,
                max_items=max_items,
            )
            for key, child_value in items[:max_items]
        ]

        if len(items) > max_items:
            children.append(
                html.Div(
                    f"... {len(items) - max_items} more keys",
                    style=leaf_style,
                )
            )

        details_kwargs = {
            "open": should_open,
        }

        if component is None:
            summary = html.Span(
                f"{name} {{ {len(items)} }}",
                style=summary_style,
            )

        else:
            component_path = component["path"]

            details_kwargs["id"] = {
                "type": "component-tree-node",
                "path": component_path,
            }

            details_kwargs["style"] = component_tree_style(
                selected=(component_path == selected_path)
            )

            summary = html.Span(
                [
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
                            "marginRight": "8px",
                        },
                        inputStyle={
                            "marginRight": "5px",
                        },
                    ),
                    html.Span(
                        f"{{ {len(items)} }}",
                        style={
                            "fontFamily": "monospace",
                            "color": "#666",
                        },
                    ),
                ]
            )

        return html.Details(
            [
                html.Summary(summary),
                html.Div(children, style=children_style),
            ],
            **details_kwargs,
        )

    # -----------------------------------------------------
    # JSON list
    # -----------------------------------------------------
    if isinstance(value, list):
        if name in {"children", "attributes"}:
            children = [
                json_tree_node(
                    name=f"[{index}]",
                    value=item,
                    json_path=json_path + [index],
                    component_lookup=component_lookup,
                    hidden_paths=hidden_paths,
                    selected_path=selected_path,
                    selected_json_path=selected_json_path,
                    max_items=max_items,
                )
                for index, item in enumerate(value[:max_items])
            ]

            if len(value) > max_items:
                children.append(
                    html.Div(
                        f"... {len(value) - max_items} more items",
                        style=leaf_style,
                    )
                )

            return html.Details(
                [
                    html.Summary(
                        f"{name} [ {len(value)} ]",
                        style=summary_style,
                    ),
                    html.Div(children, style=children_style),
                ],
                open=should_open,
            )

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

    # Scalar
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

    trace_map = {
        component["path"]: {
            "trace_indices": component["trace_indices"],
            "name": component["name"],
            "nx_class": component["nx_class"],
            "path": component["path"],
        }
        for component in components
    }

    def build_tree(tree_data, hidden_paths, selected_path=None):
        tree_components = collect_components(tree_data)

        component_lookup = {
            id(component["node"]): component
            for component in tree_components
        }

        selected_json_path = None

        for component in tree_components:
            if component["path"] == selected_path:
                selected_json_path = component["json_path"]
                break

        return json_tree_node(
            name="root",
            value=tree_data,
            json_path=[],
            component_lookup=component_lookup,
            hidden_paths=hidden_paths,
            selected_path=selected_path,
            selected_json_path=selected_json_path,
            max_items=100,
            open_node=True,
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
                id="component-trace-map",
                data=trace_map,
            ),
            dcc.Store(
                id="selected-component-path",
                data=None,
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
            # Left resizable JSON panel
            # -------------------------------------------------
            html.Div(
                [
                    html.H3(
                        "JSON structure",
                        style={"marginTop": "0"},
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
                                style={"cursor": "pointer"},
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    ),

                    html.Div(
                        "Exclude from 3D view",
                        style={"fontWeight": "bold"},
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
                                style={"cursor": "pointer"},
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    ),

                    html.Div(
                        (
                            "Edit values, then click Update geometry. "
                            "Undo/Redo restores JSON states and rebuilds "
                            "the plot. Numeric fields accept integers, "
                            "decimals and scientific notation. "
                            "Double-click a plotted point to open and "
                            "highlight its corresponding component."
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
                            data,
                            initial_hidden_paths,
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
            # Right Plotly panel
            # -------------------------------------------------
            html.Div(
                [
                    dcc.Graph(
                        id="instrument-geometry",
                        figure=figure,
                        config={
                            "responsive": True,
                            "displaylogo": False,
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
    # Save JSON through browser download dialog
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
    # Main application callback
    # ---------------------------------------------------------
    @app.callback(
        Output("instrument-geometry", "figure"),
        Output("json-data-store", "data"),
        Output("undo-history-store", "data"),
        Output("redo-history-store", "data"),
        Output("component-trace-map", "data"),
        Output("json-tree", "children"),
        Output("exclude-patterns", "value"),
        Output("selected-component-path", "data"),
        Output("plot-click-state", "data"),

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
        Input("instrument-geometry", "clickData"),

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
        State("instrument-geometry", "figure"),
        State("component-trace-map", "data"),
        State("selected-component-path", "data"),
        State("plot-click-state", "data"),

        prevent_initial_call=True,
    )
    def update_viewer(
        checkbox_values,
        edit_values,
        _apply_clicks,
        _hide_all_clicks,
        _show_all_clicks,
        _reset_clicks,
        _update_geometry_clicks,
        _undo_clicks,
        _redo_clicks,
        click_data,
        exclude_text,
        checkbox_ids,
        edit_ids,
        current_data,
        undo_history,
        redo_history,
        current_figure,
        current_trace_map,
        selected_path,
        click_state,
    ):
        triggered = ctx.triggered_id
        new_data = copy.deepcopy(current_data)

        undo_history = undo_history or []
        redo_history = redo_history or []

        if click_state is None:
            click_state = {
                "path": None,
                "time": 0.0,
            }

        new_click_state = click_state

        exclude_patterns = parse_exclude_patterns(
            exclude_text
        )

        all_paths = list(current_trace_map.keys())

        checked_paths = {
            item_id["path"]
            for values, item_id in zip(
                checkbox_values,
                checkbox_ids,
            )
            if item_id["path"] in (values or [])
        }

        # -------------------------------------------------
        # Single click: do nothing visibly.
        # Double click: open/highlight corresponding tree node.
        # -------------------------------------------------
        if triggered == "instrument-geometry":
            clicked_path = None

            if click_data and click_data.get("points"):
                clicked_path = click_data["points"][0].get(
                    "customdata"
                )

                if isinstance(clicked_path, list):
                    clicked_path = (
                        clicked_path[0]
                        if clicked_path
                        else None
                    )

            now = time.monotonic()

            is_double_click = (
                clicked_path is not None
                and clicked_path == click_state.get("path")
                and now - click_state.get("time", 0.0)
                <= DOUBLE_CLICK_SECONDS
            )

            new_click_state = {
                "path": clicked_path,
                "time": now,
            }

            if not is_double_click:
                return (
                    no_update,        # figure
                    no_update,        # JSON data
                    no_update,        # undo history
                    no_update,        # redo history
                    no_update,        # trace map
                    no_update,        # tree
                    no_update,        # exclusions
                    no_update,        # selected path
                    new_click_state,  # click state
                )

            selected_path = clicked_path

        # -------------------------------------------------
        # Apply JSON edits and save previous data in history
        # -------------------------------------------------
        if (
            isinstance(triggered, dict)
            and triggered.get("type") == "json-edit"
        ):
            previous_data = copy.deepcopy(current_data)
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
                    # Ignore invalid input until corrected.
                    pass

            if changed:
                undo_history.append(previous_data)
                undo_history = undo_history[-MAX_HISTORY_LENGTH:]

                # New edits invalidate the redo branch.
                redo_history = []

        # -------------------------------------------------
        # Undo / redo JSON state
        # -------------------------------------------------
        history_restore = False

        if triggered == "undo-json-edit":
            if undo_history:
                redo_history.append(
                    copy.deepcopy(current_data)
                )
                redo_history = redo_history[-MAX_HISTORY_LENGTH:]

                new_data = undo_history.pop()
                history_restore = True

        elif triggered == "redo-json-edit":
            if redo_history:
                undo_history.append(
                    copy.deepcopy(current_data)
                )
                undo_history = undo_history[-MAX_HISTORY_LENGTH:]

                new_data = redo_history.pop()
                history_restore = True

        # -------------------------------------------------
        # Visibility selection
        # -------------------------------------------------
        if triggered == "hide-all-components":
            visible_paths = set()

        elif triggered == "show-all-components":
            visible_paths = set()
            exclude_text = ""
            exclude_patterns = []

            # All currently known components become visible.
            visible_paths = set(all_paths)

        elif triggered == "reset-default-exclusions":
            exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
            exclude_text = ", ".join(
                DEFAULT_EXCLUDE_PATTERNS
            )

            visible_paths = {
                path
                for path, metadata in current_trace_map.items()
                if not component_matches_patterns(
                    metadata,
                    exclude_patterns,
                )
            }

        elif triggered == "apply-exclusions":
            visible_paths = {
                path
                for path, metadata in current_trace_map.items()
                if not component_matches_patterns(
                    metadata,
                    exclude_patterns,
                )
            }

        else:
            visible_paths = {
                path
                for path in checked_paths
                if path in current_trace_map
                and not component_matches_patterns(
                    current_trace_map[path],
                    exclude_patterns,
                )
            }

        # -------------------------------------------------
        # Rebuild geometry:
        # - explicit Update geometry
        # - Undo
        # - Redo
        # -------------------------------------------------
        rebuild_geometry = (
            triggered == "update-geometry"
            or triggered == "undo-json-edit"
            or triggered == "redo-json-edit"
        )

        if rebuild_geometry:
            new_figure, new_components = plot_instrument(
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

            current_trace_map = {
                component["path"]: {
                    "trace_indices": component["trace_indices"],
                    "name": component["name"],
                    "nx_class": component["nx_class"],
                    "path": component["path"],
                }
                for component in new_components
            }

            all_paths = list(current_trace_map.keys())

            visible_paths = {
                path
                for path in visible_paths
                if path in current_trace_map
            }

        # -------------------------------------------------
        # Update visibility of already-created traces
        # -------------------------------------------------
        for path, metadata in current_trace_map.items():
            trace_visible = path in visible_paths

            for trace_index in metadata["trace_indices"]:
                if trace_index < len(current_figure["data"]):
                    current_figure["data"][trace_index][
                        "visible"
                    ] = trace_visible

        hidden_paths = set(all_paths) - visible_paths

        # -------------------------------------------------
        # Preserve existing expanded/collapsed HTML Details state:
        # - ordinary JSON edits
        # - Update geometry
        #
        # Undo/redo recreates tree inputs, so it intentionally
        # rebuilds the tree in order to show restored values.
        # -------------------------------------------------
        preserve_tree_state = (
            triggered == "update-geometry"
            or (
                isinstance(triggered, dict)
                and triggered.get("type") == "json-edit"
            )
        )

        if history_restore:
            preserve_tree_state = False

        if preserve_tree_state:
            tree = no_update

        else:
            tree = build_tree(
                new_data,
                hidden_paths,
                selected_path=selected_path,
            )

        return (
            current_figure,
            new_data,
            undo_history,
            redo_history,
            current_trace_map,
            tree,
            exclude_text,
            selected_path,
            new_click_state,
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
        help=(
            "Additional initially excluded wildcard patterns. "
            "Example: --exclude 'monitor*' 'NXdetector'"
        ),
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
            "Maximum child geometry points per component "
            "sent to the browser. Default: 100000."
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